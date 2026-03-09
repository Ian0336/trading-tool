import subprocess

def get_secret(service, account):
    cmd = ["security", "find-generic-password", "-s", service, "-a", account, "-w"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None

# 直接拿到密鑰
api_key = get_secret("MyProject", "admin")