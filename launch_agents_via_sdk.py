import argparse
import os
import sys
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

    # VERTEX_AI_SA_B64 is required for Graphics Definition V2 video search (multimodal embeddings)
    vertex_ai_sa_b64 = os.environ.get("VERTEX_AI_SA_B64", "")

    # # Main ones
    # studio_name = "course-generation-agents"
    # teamspace = "Vision-model"
    # user = "dilip"

    # Niket one for testing background job
    studio_name = "latest-19-02"
    teamspace = "Vision-model"
    user = "dilip"

    print(f"[INFO] Initializing Studio '{studio_name}' in teamspace '{teamspace}'...")
    studio = Studio(name=studio_name, teamspace=teamspace, user=user, create_ok=True)

    print("[INFO] Installing 'jobs' plugin...")
    studio.install_plugin("jobs")
    jobs_plugin = studio.installed_plugins["jobs"]

    agent = args.agent_name
    
    # Export env vars for the job: GDRIVE_SA_B64 (required); VERTEX_AI_SA_B64 (for graphics_definition_v2 video search)
    export_env = f"export GDRIVE_SA_B64='{gdrive_sa_b64}' && export VERTEX_AI_SA_B64='{vertex_ai_sa_b64}' && "
    command = (
        f"echo 'numpy<2' > /tmp/constraints.txt && "
        f"pip install 'numpy<2' 'matplotlib>=3.9' 'scikit-learn>=1.5' google-cloud-aiplatform && "
        f"pip install -c /tmp/constraints.txt -r requirements.txt && "
        f"{export_env}"
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
    
    # Exit with non-zero code if job failed
    status_str = str(job.status).lower()
    if status_str not in ("succeeded", "completed", "success", "status.succeeded"):
        print(f"[ERROR] Job failed with status: {job.status}")
        sys.exit(1)

if __name__ == "__main__":
    main()