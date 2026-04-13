import base64
import io
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import (DataPart, FilePart, FileWithBytes, Message, Part, Role,
                       TaskArtifactUpdateEvent, TaskState,
                       TaskStatusUpdateEvent, TextPart)
from a2a.utils import get_message_text, new_agent_text_message
from mlebench.data import download_and_prepare_dataset
from mlebench.grade import grade_csv, validate_submission
from mlebench.registry import Competition, Registry

from messenger import Messenger


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class Agent:
    # Fill in: list of required participant roles, e.g. ["pro_debater", "con_debater"]
    required_roles: list[str] = ["agent"]
    # Fill in: list of required config keys, e.g. ["topic", "num_rounds"]
    required_config_keys: list[str] = ["competition_id"]

    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here
        self.registry = Registry()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"

    def create_competition_tar(self, competition: Competition) -> bytes:
        """Create a tar archive of the competition's public data."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            # Add all files from public_dir
            tar.add(competition.public_dir, arcname='home/data', recursive=True)

        tar_buffer.seek(0)
        return tar_buffer.read()

    async def handle_validation_request(self, message: Message, competition: Competition) -> str:
        """Handle a validation request from the agent."""
        # Extract submission file from message parts
        submission_data = None
        for part in message.parts:
            if isinstance(part.root, FilePart):
                file_data = part.root.file
                if isinstance(file_data, FileWithBytes):
                    submission_data = base64.b64decode(file_data.bytes)
                break

        if not submission_data:
            return "Error: No submission file provided"

        # Save to temp file and validate
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(submission_data)
            submission_path = Path(f.name)

        try:
            is_valid, msg = validate_submission(submission_path, competition)
            return msg
        finally:
            submission_path.unlink()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Run MLE-Bench evaluation on an agent."""
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        competition_id = request.config["competition_id"]
        agent_url = str(request.participants["agent"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting MLE-Bench evaluation for {competition_id}")
        )

        # Get competition object
        try:
            competition = self.registry.get_competition(competition_id)
        except Exception as e:
            await updater.failed(new_agent_text_message(f"Failed to load competition: {e}"))
            return

        # Clean stale artifacts from cached public data so checksums pass
        try:
            public_dir = competition.public_dir
            if public_dir.exists():
                for stale in public_dir.glob("submission.csv"):
                    stale.unlink()
        except Exception:
            pass  # best-effort; download_and_prepare_dataset will fail with a clear message if needed

        # Prepare competition data
        try:
            download_and_prepare_dataset(competition, overwrite_leaderboard=True)
        except Exception as e:
            await updater.failed(new_agent_text_message(f"Failed to prepare competition data: {e}"))
            return

        # Create competition data tar
        try:
            competition_tar = self.create_competition_tar(competition)
        except Exception as e:
            await updater.failed(new_agent_text_message(f"Failed to prepare competition data: {e}"))
            return

        # Prepare instructions for the agent
        instructions = (Path(__file__).parent / "instructions.txt").read_text()

        # Send competition to agent and handle the conversation
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Sending competition data to agent...")
        )

        try:
            result = await self.converse_with_agent(
                agent_url=agent_url,
                competition=competition,
                competition_tar=competition_tar,
                instructions=instructions,
                updater=updater,
            )
        except Exception as e:
            await updater.failed(new_agent_text_message(f"Error during evaluation: {e}"))
            return

        # Grade the final submission
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Grading submission...")
        )

        try:
            grade_report = self.grade_submission(result["submission_csv"], competition)
        except Exception as e:
            await updater.failed(new_agent_text_message(f"Failed to grade submission: {e}"))
            return

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=f"Score: {grade_report.score}")),
                Part(root=DataPart(data=grade_report.to_dict()))
            ],
            name="Result",
        )

        medal = (
            "Gold" if grade_report.gold_medal else
            "Silver" if grade_report.silver_medal else
            "Bronze" if grade_report.bronze_medal else
            "Above median" if grade_report.above_median else
            "Below median"
        )
        summary = (
            f"Competition: {competition_id}\n"
            f"Score: {grade_report.score}\n"
            f"Medal: {medal}\n"
            f"Valid submission: {grade_report.valid_submission}"
        )
        await updater.complete(new_agent_text_message(summary))

    async def converse_with_agent(
        self,
        agent_url: str,
        competition: Competition,
        competition_tar: bytes,
        instructions: str,
        updater: TaskUpdater,
    ) -> dict[str, bytes]:
        """
        Send competition to agent and handle bidirectional conversation.
        Returns final submission data with 'submission_csv' key.
        """
        import httpx
        from a2a.client import A2ACardResolver, ClientConfig, ClientFactory

        submission_csv: bytes | None = None

        async with httpx.AsyncClient(timeout=3600) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=agent_url)
            agent_card = await resolver.get_agent_card()
            config = ClientConfig(httpx_client=httpx_client, streaming=True)
            factory = ClientFactory(config)
            client = factory.create(agent_card)

            # Create initial message with tar attachment
            initial_msg = Message(
                kind="message",
                role=Role.user,
                parts=[
                    Part(root=TextPart(text=instructions)),
                    Part(root=FilePart(
                        file=FileWithBytes(
                            bytes=base64.b64encode(competition_tar).decode('ascii'),
                            name="competition.tar.gz",
                            mime_type="application/gzip"
                        )
                    ))
                ],
                message_id=uuid4().hex,
            )

            # Send message and process responses
            async for event in client.send_message(initial_msg):
                match event:
                    case (task, TaskStatusUpdateEvent() as update) if (msg := update.status.message) and "validate" in get_message_text(msg):
                        validation_result = await self.handle_validation_request(msg, competition)
                        response_msg = Message(
                            kind="message",
                            role=Role.user,
                            parts=[Part(root=TextPart(text=validation_result))],
                            message_id=uuid4().hex,
                            context_id=task.context_id,
                        )
                        async for _ in client.send_message(response_msg):
                            pass  # Wait for ack

                    case (task, TaskArtifactUpdateEvent() as artifact_event):
                        for part in artifact_event.artifact.parts:
                            if isinstance(part.root, FilePart):
                                file_data = part.root.file
                                if isinstance(file_data, FileWithBytes):
                                    submission_csv = base64.b64decode(file_data.bytes)

                    case (_, TaskStatusUpdateEvent()):
                        pass  # Normal task state transitions (created, working, completed) — no action needed

                    case _:
                        import logging
                        logging.getLogger(__name__).warning("Purple agent event (truly unhandled): %s", type(event))

        if not submission_csv:
            raise ValueError("Agent did not submit a valid submission.csv")

        return {"submission_csv": submission_csv}

    def grade_submission(self, submission_csv: bytes, competition: Competition):
        """Grade a submission CSV."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(submission_csv)
            submission_path = Path(f.name)

        try:
            return grade_csv(submission_path, competition)
        finally:
            submission_path.unlink()
