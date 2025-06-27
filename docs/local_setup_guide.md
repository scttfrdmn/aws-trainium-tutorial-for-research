# Local System Prerequisites and Setup Guide

This guide covers the complete setup process for running the AWS Trainium & Inferentia tutorial on your local development machine across different operating systems.

## Overview

Before running ML workloads on AWS Trainium and Inferentia, you need to set up your local development environment for:
- AWS CLI and credentials configuration
- Python development environment
- Tutorial dependencies and tools
- Optional: Local testing and development utilities

## Prerequisites by Operating System

### 🍎 macOS Setup

#### System Requirements
- macOS 10.15 (Catalina) or later
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space
- Internet connection for downloads

#### Required Software

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.8+**:
   ```bash
   # Using Homebrew (recommended)
   brew install python@3.11

   # Verify installation
   python3 --version  # Should show 3.8+ or higher
   ```

3. **Install Git**:
   ```bash
   brew install git
   ```

4. **Install AWS CLI v2**:
   ```bash
   # Download and install
   curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
   sudo installer -pkg AWSCLIV2.pkg -target /

   # Verify installation
   aws --version
   ```

5. **Install Additional Development Tools**:
   ```bash
   # Essential build tools
   xcode-select --install

   # Package management
   brew install curl wget
   ```

#### macOS-Specific Configuration

1. **Configure Python PATH** (add to `~/.zshrc` or `~/.bash_profile`):
   ```bash
   export PATH="/opt/homebrew/bin:$PATH"  # For Apple Silicon Macs
   export PATH="/usr/local/bin:$PATH"     # For Intel Macs
   ```

2. **Set up virtual environment**:
   ```bash
   python3 -m venv ~/venv/trainium-tutorial
   source ~/venv/trainium-tutorial/bin/activate
   ```

---

### 🪟 Windows 11 Setup

#### System Requirements
- Windows 11 (or Windows 10 version 1903+)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space
- PowerShell 5.1+ or PowerShell Core 7+

#### Option A: Windows Subsystem for Linux (Recommended)

1. **Enable WSL2**:
   ```powershell
   # Run PowerShell as Administrator
   wsl --install
   # Restart computer when prompted
   ```

2. **Install Ubuntu**:
   ```powershell
   wsl --install -d Ubuntu
   ```

3. **Follow Linux setup instructions** (see Ubuntu section below) within WSL2.

#### Option B: Native Windows Setup

1. **Install Python 3.8+**:
   - Download from [python.org](https://www.python.org/downloads/windows/)
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install for all users"

   ```powershell
   # Verify installation
   python --version
   ```

2. **Install Git for Windows**:
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use default settings during installation

3. **Install AWS CLI v2**:
   ```powershell
   # Download and run installer
   Invoke-WebRequest -Uri "https://awscli.amazonaws.com/AWSCLIV2.msi" -OutFile "AWSCLIV2.msi"
   Start-Process msiexec.exe -ArgumentList "/i AWSCLIV2.msi /quiet" -Wait

   # Verify installation (restart PowerShell first)
   aws --version
   ```

4. **Install Visual Studio Build Tools** (for Python packages):
   - Download "Build Tools for Visual Studio 2022"
   - Select "C++ build tools" workload

#### Windows-Specific Configuration

1. **Set up PowerShell execution policy**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Configure virtual environment**:
   ```powershell
   python -m venv %USERPROFILE%\venv\trainium-tutorial
   %USERPROFILE%\venv\trainium-tutorial\Scripts\activate.bat
   ```

---

### 🐧 Linux (Ubuntu/Debian) Setup

#### System Requirements
- Ubuntu 20.04+ or Debian 11+
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

#### Package Installation

1. **Update system packages**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Python 3.8+**:
   ```bash
   sudo apt install -y python3 python3-pip python3-venv python3-dev

   # Verify installation
   python3 --version
   ```

3. **Install development essentials**:
   ```bash
   sudo apt install -y build-essential curl wget git software-properties-common
   ```

4. **Install AWS CLI v2**:
   ```bash
   # Download and install
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install

   # Verify installation
   aws --version
   ```

#### Linux-Specific Configuration

1. **Set up virtual environment**:
   ```bash
   python3 -m venv ~/venv/trainium-tutorial
   source ~/venv/trainium-tutorial/bin/activate
   ```

2. **Configure pip (optional but recommended)**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

---

## Common Setup Steps (All Platforms)

### 1. AWS Configuration

1. **Configure AWS credentials**:
   ```bash
   aws configure
   ```
   Enter:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region: `us-east-1` (recommended for Trainium)
   - Output format: `json`

2. **Verify AWS access**:
   ```bash
   aws sts get-caller-identity
   ```

### 2. Clone and Setup Tutorial

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/aws-trainium-tutorial-for-research.git
   cd aws-trainium-tutorial-for-research
   ```

2. **Install Python dependencies**:
   ```bash
   # Activate virtual environment first
   pip install -e .[dev,science]
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### 3. Run Environment Checker

1. **Check AWS environment**:
   ```bash
   python scripts/aws_environment_checker.py
   ```

2. **Auto-fix issues** (optional):
   ```bash
   python scripts/aws_environment_checker.py --auto-fix
   ```

### 4. Verify Installation

1. **Run tests**:
   ```bash
   make test
   ```

2. **Check linting**:
   ```bash
   make lint
   ```

3. **Test AWS connectivity**:
   ```bash
   python -c "import boto3; print('✅ AWS SDK working'); print(f'Region: {boto3.Session().region_name}')"
   ```

---

## Troubleshooting

### Common Issues

#### Permission Errors (All Platforms)
```bash
# If pip fails with permission errors
pip install --user -e .[dev,science]
```

#### Python Version Issues
```bash
# Check Python version
python --version || python3 --version

# If version is < 3.8, install newer version
# Follow OS-specific Python installation above
```

#### AWS Credentials Issues
```bash
# Check current configuration
aws configure list

# Reconfigure if needed
aws configure

# Test with minimal permissions
aws sts get-caller-identity
```

#### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf ~/venv/trainium-tutorial  # or %USERPROFILE%\venv\trainium-tutorial on Windows
python3 -m venv ~/venv/trainium-tutorial
source ~/venv/trainium-tutorial/bin/activate  # or activate.bat on Windows
```

### Platform-Specific Issues

#### macOS: Command Line Tools
```bash
# If xcode tools missing
xcode-select --install
```

#### Windows: Long Path Issues
```powershell
# Enable long path support (requires admin)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### Linux: Missing Development Headers
```bash
# Install additional development packages
sudo apt install -y python3-dev libffi-dev libssl-dev
```

---

## Advanced Configuration

### Development Environment Enhancements

1. **IDE Setup**:
   - **VS Code**: Install Python extension, AWS Toolkit
   - **PyCharm**: Configure AWS integration
   - **Vim/Neovim**: Install Python LSP, AWS CLI completion

2. **Shell Completion**:
   ```bash
   # AWS CLI completion
   echo 'complete -C aws_completer aws' >> ~/.bashrc  # or ~/.zshrc
   ```

3. **Environment Variables**:
   ```bash
   # Add to ~/.bashrc, ~/.zshrc, or equivalent
   export AWS_DEFAULT_REGION=us-east-1
   export AWS_PAGER=""  # Disable pager for CLI output
   ```

### Optional: Local Testing with Docker

1. **Install Docker**:
   - macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Windows: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Linux: `sudo apt install docker.io docker-compose`

2. **Test environment**:
   ```bash
   # Build test container
   docker build -t trainium-tutorial .

   # Run tests in container
   docker run --rm trainium-tutorial make test
   ```

---

## Next Steps

After completing this setup:

1. ✅ **Verify environment**: Run `python scripts/aws_environment_checker.py`
2. ✅ **Review AWS costs**: Understand pricing in `docs/cost_analysis.md`
3. ✅ **Start with examples**: Begin with `examples/basic/hello_trainium.py`
4. ✅ **Join community**: Check `CONTRIBUTING.md` for contribution guidelines

## Support

- 📖 **Documentation**: See `docs/` directory
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Discussions**: Use GitHub Discussions for questions
- 📧 **Contact**: scott.friedman@example.com

---

*Last updated: 2024-12-19*
*Tested on: macOS 14, Windows 11, Ubuntu 22.04*
