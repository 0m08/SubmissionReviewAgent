# Setting Up a Public Link on Lightning.ai to Run Agents on Streamlit Cloud

This guide will walk you through setting up your project on [Lightning.ai](https://lightning.ai) to run agents using Streamlit. Follow these steps to get started!

---

## Step 1: Sign Up & Set Up Your Lightning.ai Studio

1. Go to [https://lightning.ai](https://lightning.ai) and **sign up** for a free account.
2. Once logged in, you’ll be **onboarded automatically**. Follow the onboarding steps.
3. This process will create your **free Studio**, an online cloud-based IDE similar to Visual Studio Code.

---

## Step 2: Sync Your GitHub Repository

Before cloning your code into Lightning.ai, ensure your GitHub branches are synced:

- Ensure **all branches are merged into `main`**.
- No branch should be **ahead** the `main` branch.
- Use GitHub to verify and complete merges if needed.

---

## Step 3: Open Terminal in Lightning Studio and Clone the Repository

In your Lightning.ai Studio:

1. Open the **Terminal** tab.
2. Initialize GitHub and clone your repository:
   - Authenticate GitHub using the CLI:
     ```bash
     gh auth login
     ```
   - Get the GitHub CLI clone URL from your repo:
     For example:
     ```bash
     gh repo clone username/repo-name
     ```
   - Run this command in the terminal to clone the repo.

---

## Step 4: Move the Agent Files to the Main Workspace

1. Your agents will be inside the `course_generation` folder.
2. Copy the contents of this folder into the **main workspace folder** (called `THIS STUDIO`).
3. **Delete** the `course_generation` folder to avoid Git conflicts and broken links in source control.
4. **Rename** the folder if necessary to organize your workspace.

---

## Step 5: Install All Dependencies

In the Terminal:

1. Install packages from `requirements.txt`
   ```bash
   pip install -r requirements.txt
   ```

2. Install **Playwright**
   ```bash
   playwright install
   ```
3. Install system packages from `packages.txt`
  ```bash
sudo apt update && sudo xargs -a packages.txt apt install -y
```
---

## Step 6: Add Secrets and API Keys

1. Open the **"Vision Model"** Studio page in Lightning.ai.
2. Navigate to **Settings** — this opens in a new window.
3. On the left sidebar, click on **Secrets**.
4. For each key-value pair in your local `.env` file:
   - Click **"Create New Secret"**
   - Add the **name** and **value** (without double quotes `" "`).
5. Add all necessary credentials and API keys here. These will be securely injected into your app during runtime.

### Getting the Environment Variable for the Service Account
1. To get the environment variable for the service account, we convert the json file into base64 and then pass it as env variable.
2. Then, within the code, we decode the base64 to get json back which is used for authentication.
3. Here's the code to convert to base64 -
   ```bash
   import base64, pathlib, json, sys, textwrap
   data = pathlib.Path("/content/service-credentials.json").read_bytes()
   print(base64.b64encode(data).decode())
   ```

---



## Step 7: Add the Streamlit Plugin

1. On the right panel of your Studio interface, click the **Add** button.
2. In the **"Install Studio Plugins"** window:
   - Scroll to the **Web Apps** section.
   - If **Streamlit** is not visible, use the **search bar** to find it.
   - Click **Install** to add it as a plugin.
3. Once installed, the **Streamlit plugin** will appear on the far right panel of your Studio.

---

## Step 8: Connect to the Streamlit Plugin

1. Click on the **Streamlit** plugin icon, and add a new app.
2. To start the new app, ensure that the current directory is chosen, for example, we are using `streamlit_app.py` to run  the agents.
3. Choose it on the left side with the files and click on **Auto Start**, then run to create the app.
4. The following command should be auto-generated:
   ```bash
   streamlit run --server.port 8501 streamlit_app.py
   ```
5. Run this command to launch your Streamlit app within Lightning.ai.

6. Test to ensure the UI is running correctly and that the public link works.

---

## Best Practices

- Always ensure that the `.env` file is added to your `.gitignore` to prevent pushing sensitive information to GitHub.
- Always store API keys and credentials securely using the **Secrets** section in Lightning.ai.
- Keep your GitHub repository **in sync**, especially the `main` branch, to avoid deployment issues.

---

# Running Agents via the Lightning SDK (`launch_agents_via_sdk.py`)

You can launch any agent as a background job on Lightning.ai using the provided SDK script. This is useful for automated, headless, or batch runs from your VSCode/Cursor terminal.

## 1. Prerequisites

- Ensure all environment variables are set in your .env file.
- All dependencies should be installed (see previous steps).

---

## 2. Usage

Open a terminal in your VSCode/Cursor terminal and run the following command:

```bash
python launch_agents_via_sdk.py --sheet_link "ENTER_GOOGLE_SHEET_LINK" --drive_folder_id "ENTER THE GOOGLE_DRIVE_FOLDER_ID" --agent_name "ENTER_THE_AGENT_NAME_THAT_YOU_WANT_TO_ RUN"
```

### Arguments

- `--sheet_link`  
  The full URL to your Google Sheet (where agent logs and data will be stored).
- `--drive_folder_id`  
  The Google Drive folder ID for your course.
- `--agent_name`  
  The name of the agent you want to run.  
  **Examples:** `course_outline`, `research_notes`, `slide_chunks`, `graphics_definition`, `assessment`

### Example

```bash
python launch_agents_via_sdk.py --sheet_link "https://docs.google.com/spreadsheets/d/your-sheet-id" --drive_folder_id "1JDIuQnP7B4KznzO9FZAKOhDG3sdMJnHi" --agent_name "slide_chunks"
```

---

## 3. What Happens

- The script will submit a job to Lightning.ai to run the specified agent.
- Logs will be printed in the terminal, and you can monitor job status in the same,terminal.
- A link to the Job logs on Lightning.ai will also be generated in the terminal.
- The agent will create and update the "Agent logs" sheet in your Google Sheet, just like the UI-based run along with the other outputs that are generated by the agent.
- You will get to know that the agent has completed running all it's steps when you see this message in your terminal - "Job 'job_name' completed with status: Completed"

---

# Running Agents via Batch Jobs in Lightning.ai Studio (`run_agent_cli.py`)

You can run any agent as a batch job directly in the Lightning.ai Studio using the `run_agent_cli.py` script.

## 1. Prerequisites

- Ensure all environment variables are set in your Lightning.ai Studio secrets.
- All dependencies should be installed in your Studio environment (see previous steps).

---

## 2. Usage

1. Go to the **Batch Jobs** section in your Lightning.ai Studio.
2. Click the **+ New Job** button.
3. In the Bash command field, enter the following command:

```bash
python run_agent_cli.py --sheet_link "ENTER GOOGLE_SHEET_LINK" --drive_folder_id "ENTER THE_GOOGLE_DRIVE_FOLDER_ID" --agent_name "ENTER THE AGENT NAME THAT YOU WANT TO RUN"
```

### Arguments

- `--sheet_link`  
  The full URL to your Google Sheet (where agent logs and data will be stored).
- `--drive_folder_id`  
  The Google Drive folder ID for your course.
- `--agent_name`  
  The name of the agent you want to run.  
  **Examples:** `course_outline`, `research_notes`, `slide_chunks`, `graphics_definition`, `assessment`

### Example

```bash
python run_agent_cli.py --sheet_link "https://docs.google.com/spreadsheets/d/your-sheet-id" --drive_folder_id "1JDIuQnP7B4KznzO9FZAKOhDG3sdMJnHi" --agent_name "slide_chunks"
```

---

## 3. What Happens

- The script will run as a batch job in the Lightning.ai Studio cloud environment.
- Logs and job status will be available in the Batch Jobs section on Lightning.ai.
- - The agent will create and update the "Agent logs" sheet in your Google Sheet, just like the UI-based run along with the other outputs that are generated by the agent.

---










