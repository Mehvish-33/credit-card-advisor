import bcrypt

SALT = b"my_static_salt"

def hash_password(password: str):
    return bcrypt.hashpw(password.encode() + SALT, bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str):
    return bcrypt.checkpw(password.encode() + SALT, hashed.encode())
