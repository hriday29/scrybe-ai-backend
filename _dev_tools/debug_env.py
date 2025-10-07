import os
from dotenv import load_dotenv

print("--- Starting Environment Variable Debugger ---")

# Get the current working directory and the expected path of the .env file
current_directory = os.getcwd()
env_path = os.path.join(current_directory, '.env')

print(f"Current Directory: {current_directory}")
print(f"Expecting to find .env file at: {env_path}")

# Check if the .env file actually exists at that path
if not os.path.exists(env_path):
    print("\n❌ CRITICAL: The .env file was NOT FOUND in the directory you are running this script from.")
else:
    print("\n✅ OK: .env file found.")
    
    # Try to load the file
    load_dotenv()

    # Now, try to read the specific variable we are looking for
    totp_secret = os.getenv('ANGELONE_TOTP_SECRET')

    print("\n--- Checking for ANGELONE_TOTP_SECRET ---")
    if totp_secret:
        print(f"✅ SUCCESS: Variable was loaded correctly.")
        print(f"   Value starts with: '{totp_secret[:4]}...' and ends with: '...{totp_secret[-4:]}'")
    else:
        print("❌ FAILED: os.getenv('ANGELONE_TOTP_SECRET') returned nothing.")
        print("   This confirms the variable is not being parsed correctly from your .env file.")
        print("   This is almost always due to a syntax error (like an unclosed quote) on a line *above* it in the file.")
        
    print("\n--- Forcing a raw read of the file to check for errors ---")
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print("Raw file content:")
        for i, line in enumerate(lines):
            print(f"  Line {i+1}: {line.strip()}")
    except Exception as e:
        print(f"Could not even read the file manually. This points to a file encoding issue. Error: {e}")