from fastapi import FastAPI, HTTPException, Depends
from backend.models.user import UserSignup, UserLogin
from backend.utils.hash import hash_password, verify_password
from backend.database.mongo import users

app = FastAPI()

@app.post("/signup")
def signup(user: UserSignup):
    if users.find_one({"email": user.email}):
        raise HTTPException(400, "User already exists")
    hashed = hash_password(user.password)
    users.insert_one({"email": user.email, "password": hashed})
    return {"msg": "Signup successful"}

@app.post("/login")
def login(user: UserLogin):
    existing = users.find_one({"email": user.email})
    if not existing or not verify_password(user.password, existing["password"]):
        raise HTTPException(401, "Invalid credentials")
    return {"msg": "Login success"}
