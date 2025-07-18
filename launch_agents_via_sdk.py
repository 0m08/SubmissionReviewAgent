import argparse
import os
from lightning_sdk import Machine, Studio
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Launch a Lightning AI job for a single agent via SDK.")
    parser.add_argument('--sheet_link', required=True, help='Google Sheet URL')
    parser.add_argument('--drive_folder_id', required=True, help='Google Drive folder ID')
    parser.add_argument('--agent_name', required=True, help='Name of the agent to run')
    args = parser.parse_args()

    gdrive_sa_json = os.environ.get("GDRIVE_SA_JSON")

    studio_name = "course-generation-agents"
    teamspace = "Vision-model"
    user = "dilip"

    print(f"[INFO] Initializing Studio '{studio_name}' in teamspace '{teamspace}'...")
    studio = Studio(name=studio_name, teamspace=teamspace, user=user, create_ok=True)

    print("[INFO] Installing 'jobs' plugin...")
    studio.install_plugin("jobs")
    jobs_plugin = studio.installed_plugins["jobs"]

    agent = args.agent_name
    command = (
        f"export GDRIVE_SA_JSON='{gdrive_sa_json}' && "
        f"python run_agent_cli.py "
        f"--sheet_link '{args.sheet_link}' "
        f"--drive_folder_id '{args.drive_folder_id}' "
        f"--agent_name '{agent}'"
    )

    print(f"\n[INFO] Submitting job for agent: {agent}")
    job = jobs_plugin.run(
        command,
        name = f"{agent}-job",
        machine = Machine.CPU,
        interruptible = True
    )

    print(f"[INFO] Job '{job.name}' submitted. Waiting for it to finish...")
    job.wait()
    print(f"[INFO] Job '{job.name}' completed with status: {job.status}")

if __name__ == "__main__":
    main()
