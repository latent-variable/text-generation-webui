import PyKCS11
from PyKCS11 import PyKCS11Error
from getpass import getpass

def get_username_from_smartcard():
    # Initialize the PKCS#11 library
    pkcs11 = PyKCS11.PyKCS11Lib()
    
    # Prompt for the library path
    lib_path = input("Enter the full path to the PKCS#11 library (e.g., C:\\path\\to\\pkcs11.dll): ")
    
    try:
        pkcs11.load(lib_path)
    except PyKCS11Error as e:
        print(f"Error loading PKCS#11 library: {str(e)}")
        return None
    
    # Get available slots (card readers)
    try:
        slots = pkcs11.getSlotList()
    except PyKCS11Error as e:
        print(f"Error getting slot list: {str(e)}")
        return None
    
    if not slots:
        print("No card reader found.")
        return None
    
    # Use the first available slot
    slot = slots[0]
    
    try:
        # Open a session
        session = pkcs11.openSession(slot)
        
        # Prompt for PIN
        pin = getpass("Enter your PIN: ")
        
        # Login with the PIN
        session.login(pin)
        
        # Try to find and read the CKA_LABEL attribute, which often contains the username
        objects = session.findObjects([(PyKCS11.CKA_CLASS, PyKCS11.CKO_CERTIFICATE)])
        
        for obj in objects:
            try:
                label = session.getAttributeValue(obj, [PyKCS11.CKA_LABEL])[0]
                if label:
                    return label.decode('utf-8')
            except PyKCS11Error:
                continue
        
        print("Username not found in certificate labels.")
        return None
    
    except PyKCS11Error as e:
        print(f"Error: {str(e)}")
        return None
    
    finally:
        # Always close the session
        if 'session' in locals():
            session.logout()
            session.closeSession()

# Usage
username = get_username_from_smartcard()
if username:
    print(f"Extracted username: {username}")
else:
    print("Failed to extract username")