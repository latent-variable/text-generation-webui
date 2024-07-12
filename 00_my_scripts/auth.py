# import PyKCS11
# import statistics
# from cryptography import x509
# from cryptography.hazmat.backends import default_backend



# def get_certificates_from_smartcard():
#     # Load the PKCS#11 library for OpenSC
#     lib = 'C:\\Program Files\\OpenSC Project\\OpenSC\\pkcs11\\opensc-pkcs11.dll'
#     pkcs11 = PyKCS11.PyKCS11Lib()
#     pkcs11.load(lib)
    
#     # Get available slots and open a session
#     slots = pkcs11.getSlotList()
#     if not slots:
#         raise Exception("No slots found")

#     session = pkcs11.openSession(slots[0])

#     # Authenticate with the smartcard (using PIN, for example)
#     session.login('49915991')

#     # Find the certificate object
#     certs = session.findObjects([(PyKCS11.CKA_CLASS, PyKCS11.CKO_CERTIFICATE)])
    
#     decoded_certs = []
#     for cert in certs:
#         cert_der = session.getAttributeValue(cert, [PyKCS11.CKA_VALUE])[0]
#         cert_der = bytes(cert_der)
#         cert = x509.load_der_x509_certificate(cert_der, default_backend())
#         decoded_certs.append(cert)

#     # Don't forget to logout and close the session
#     session.logout()
#     session.closeSession()
    
#     return decoded_certs

# def get_username_from_certificates():
#     cert_mode = []
#     for cert in get_certificates_from_smartcard():
#         # Convert the cert.subject to a list
#         components = list(cert.subject)
        
#         # Get the last component
#         last_component = components[-1]

#         cert_mode.append(last_component.value)
    
   
#     if len(cert_mode):
#         last_first_id = statistics.mode(cert_mode)
#         # Return the username ('LAST.FIRST.ID') from certificate subject 
#         return last_first_id
#     else:
#         # If no certificates were found, throw an exception
        


# print(get_username_from_certificates())



# import gradio as gr
# import PyKCS11
# from cryptography import x509
# from cryptography.hazmat.backends import default_backend

# # Function to interact with the smartcard
# def read_smartcard(pin):
#     # Load the PKCS#11 library for OpenSC
#     lib = 'C:\\Program Files\\OpenSC Project\\OpenSC\\pkcs11\\opensc-pkcs11.dll'
#     pkcs11 = PyKCS11.PyKCS11Lib()
#     pkcs11.load(lib)
    
#     # Get available slots and open a session
#     slots = pkcs11.getSlotList()
#     if not slots:
#         return "No slots found"
    
#     session = pkcs11.openSession(slots[0])
    
#     try:
#         # Authenticate with the smartcard using the entered PIN
#         session.login(pin)
        
#         # Find the certificate object
#         certs = session.findObjects([(PyKCS11.CKA_CLASS, PyKCS11.CKO_CERTIFICATE)])
#         cert_info = []
        
#         for cert in certs:
#             cert_der = session.getAttributeValue(cert, [PyKCS11.CKA_VALUE])[0]
#             cert_der = bytes(cert_der)
#             cert = x509.load_der_x509_certificate(cert_der, default_backend())
#             cert_info.append(f"Subject: {cert.subject}, Issuer: {cert.issuer}")
        
#         return "\n".join(cert_info)
#     except Exception as e:
#         return f"An error occurred: {e}"
#     finally:
#         # Logout and close the session
#         session.logout()
#         session.closeSession()

# # Create the Gradio interface
# iface = gr.Interface(
#     fn=read_smartcard,  # The function to call
#     inputs=gr.Textbox(label="Enter your smartcard PIN", type="password"),
#     outputs="text",
#     title="Smartcard Reader",
#     description="Enter your PIN to read the smartcard certificate information."
# )

# # Launch the interface
# iface.launch()

# import PyKCS11
# from datetime import datetime
# from cryptography import x509
# from cryptography.hazmat.primitives import hashes
# from cryptography.hazmat.backends import default_backend

# # Load the PKCS#11 library for OpenSC
# lib = 'C:\\Program Files\\OpenSC Project\\OpenSC\\pkcs11\\opensc-pkcs11.dll'

# pkcs11 = PyKCS11.PyKCS11Lib()
# pkcs11.load(lib)

# def get_smart_card_cert():
#     # Get available slots
#     slots = pkcs11.getSlotList()
#     if not slots:
#         raise Exception("No smart card reader found")

#     # Use the first slot
#     session = pkcs11.openSession(slots[0])

#     # Find objects of class CKO_CERTIFICATE
#     cert_objs = session.findObjects([(PyKCS11.CKA_CLASS, PyKCS11.CKO_CERTIFICATE)])
#     if not cert_objs:
#         raise Exception("No certificate found on smart card")

#     # Get the first certificate
#     cert_obj = cert_objs[0]
#     cert_der = bytes(session.getAttributeValue(cert_obj, [PyKCS11.CKA_VALUE])[0])

#     # Parse the certificate
#     cert = x509.load_der_x509_certificate(cert_der, default_backend())
#     return cert

# def authenticate_with_smart_card():
#     try:
#         cert = get_smart_card_cert()
        
#         # Here you would typically verify the certificate
#         # For example, check if it's issued by a trusted CA, not expired, etc.
#         if cert.not_valid_before <= datetime.now() <= cert.not_valid_after:
#             # Extract user information from the certificate
#             subject = cert.subject
#             common_name = subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
#             return {"status": "success", "user": common_name}
#         else:
#             return {"status": "error", "message": "Invalid certificate"}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# # Example usage
# result = authenticate_with_smart_card()
# print(result)


