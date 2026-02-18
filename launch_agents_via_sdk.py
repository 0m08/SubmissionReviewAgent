import argparse
import os
import base64
from lightning_sdk import Machine, Studio
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Launch a Lightning AI job for a single agent via SDK.")
    parser.add_argument('--sheet_link', required=True, help='Google Sheet URL')
    parser.add_argument('--drive_folder_id', required=True, help='Google Drive folder ID')
    parser.add_argument('--agent_name', required=True, help='Name of the agent to run')
    args = parser.parse_args()

    # Use GDRIVE_SA_B64 directly if available, otherwise encode GDRIVE_SA_JSON
    gdrive_sa_b64 = os.environ.get("GDRIVE_SA_B64")
    if not gdrive_sa_b64:
        gdrive_sa_json = os.environ.get("GDRIVE_SA_JSON")
        if gdrive_sa_json:
            gdrive_sa_b64 = base64.b64encode(gdrive_sa_json.encode()).decode()
        else:
            gdrive_sa_b64 = ""
    
    # Main ones
    studio_name = "course-generation-agents"
    teamspace = "Vision-model"
    user = "dilip"

    # # Niket one for testing background job
    # studio_name = "simple-coffee-ezuw"
    # teamspace = "deploy-model-project"
    # user = "niket4204"

    print(f"[INFO] Initializing Studio '{studio_name}' in teamspace '{teamspace}'...")
    studio = Studio(name=studio_name, teamspace=teamspace, user=user, create_ok=True)

    print("[INFO] Installing 'jobs' plugin...")
    studio.install_plugin("jobs")
    jobs_plugin = studio.installed_plugins["jobs"]

    agent = args.agent_name
    command = (
        f"export GDRIVE_SA_B64='{gdrive_sa_b64}' && "
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