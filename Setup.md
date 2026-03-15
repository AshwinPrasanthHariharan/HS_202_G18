# Setup Instructions

This guide will help you set up your development environment for the HS202 project. The setup involves installing Pixi (a fast package manager), cloning the repository, and configuring the project environment. Follow these steps in order.

## Prerequisites

### For Windows Users: Setting Up WSL

If you're using Windows, you'll need Windows Subsystem for Linux (WSL) to run Linux commands. WSL lets you run a full Linux environment on your Windows machine.

**Why do you need this?** The project uses Linux-based tools and commands that don't work natively on Windows.

1. **Open PowerShell or Command Prompt as Administrator**
   - Right-click on the Start menu and select "Windows PowerShell (Admin)" or "Command Prompt (Admin)"

2. **Install WSL by running this command:**

   ```powershell
   wsl --install
   ```

   **What this does:**
   - Enables WSL on your system
   - Downloads and installs Ubuntu (a popular Linux distribution)
   - Sets Ubuntu as your default WSL environment

3. **Restart your computer** if prompted to complete the installation.

4. **Set up your Linux user account:**
   - After restart, WSL will open automatically
   - Create a username and password when prompted
   - Remember this password - you'll need it for administrative tasks

5. **Access your Linux environment:**
   - Open the Start menu and search for "Ubuntu"
   - Or open PowerShell/Command Prompt and type `wsl`

6. **Continue with the rest of the setup** in your WSL terminal.

**Troubleshooting:** If you have issues, visit the [official WSL documentation](https://docs.microsoft.com/en-us/windows/wsl/install) for detailed help.

## Installing Pixi

Pixi is a modern package manager that handles Python environments and dependencies. It's faster and simpler than traditional tools like conda or pip.

**Why Pixi?** It automatically manages virtual environments and dependencies, making setup consistent across different machines.

1. **Install Pixi by running:**

   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

   **What this does:** Downloads and installs Pixi on your system.

2. **Make Pixi available in your terminal:**
   - Restart your terminal, or
   - Run `source ~/.bashrc` (or your shell's config file)

3. **Verify installation:**
   - Run `pixi --version` to confirm it's working

## Setting Up the Environment

Now we'll get the project code and set up its dependencies.

1. **Clone the project repository:**

   ```bash
   git clone git@github.com:AshwinPrasanthHariharan/HS202_G18.git
   cd HS202_G18
   ```

   **What this does:** Downloads all project files and navigates into the project directory.

2. **Install project dependencies:**

   ```bash
   pixi install
   ```

   **What this does:**
   - Creates a virtual environment for the project
   - Installs Python and all required packages (as defined in `pixi.toml`)
   - This may take a few minutes the first time

3. **Activate the environment (optional):**

   ```bash
   pixi shell
   ```

   **Why optional?** Pixi can run commands directly in the project environment without activation. Use `exit` to leave the shell when done.

## Verification

To make sure everything is set up correctly:

1. **Check that you're in the right directory:**
   ```bash
   pwd
   ```
   Should show something like `/home/yourname/HS202_G18`

2. **Verify Pixi environment:**
   ```bash
   pixi list
   ```
   Should show installed packages

3. **Test Python:**
   ```bash
   python --version
   ```
   Should show Python 3.11 or higher

## Working with the Dataset

This project uses Google Drive to store and share the dataset. The repository includes two Python scripts to help you synchronize the dataset:

### Downloading the Dataset (`gd_pull`)

To download the latest dataset from Google Drive:

```bash
python gd_pull.py
```

**What this does:**
- Downloads a zip file containing the dataset from Google Drive
- Extracts the contents to the `dataset/` directory
- Removes the downloaded zip file
- Overwrites any existing local dataset

**When to use:** Run this when you first set up the project, or when you need to get updates to the dataset.

### Uploading Dataset Changes (`gd_push`)

To prepare your local dataset changes for upload to Google Drive:

```bash
python gd_push.py
```

**What this does:**
- Creates a zip archive of your `dataset/` directory
- Calculates a SHA256 checksum of the zip file
- Saves the checksum to `dataset.sha256`
- Instructs you to manually upload the zip file to Google Drive

**Upload location:** [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1CjpIhF7LBYsFMFngy_TTkUxclgdYSyXX?usp=sharing)

**When to use:** Run this after making changes to the dataset that you want to share with others.

**Important notes:**
- After running `gd_push.py`, you need to manually upload `dataset.zip` to the Google Drive location
- The `dataset.sha256` file helps verify the integrity of the uploaded file
- Make sure you have the `gdown` Python package installed (included in the Pixi environment)

## Additional Notes

- **Python requirement:** Make sure you have Python 3.11+ installed before installing Pixi
- **Git setup:** Ensure you have Git installed and configured with SSH keys for GitHub access
- **Troubleshooting:** If you encounter issues, check the [Pixi documentation](https://pixi.sh/) or create an issue in the repository
- **Updates:** To update dependencies later, run `pixi install` again after pulling repository changes
