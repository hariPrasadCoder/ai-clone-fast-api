from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import boto3
import os
import json
import csv
import io
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AWS clients
session = boto3.Session(region_name="us-east-1")
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_7JLSus_U3m4SxY6snjBuCB5KAqBXGMm2h5YrZkSicYdCqQhVwDCNVGrybyf26H8MQVDwTa")
pinecone_index = pc.Index("user-posts")

# Configuration
BUCKET_NAME = "user-linkedin-posts-data"
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

class UploadRequest(BaseModel):
    user_email: str
    linkedin_posts: str

class CSVUploadRequest(BaseModel):
    user_email: str
    linkedin_posts_csv: str

def process_and_store(user_email: str, text: str):
    """Background task handling both S3 upload and Pinecone storage"""
    try:
        # 1. Upload to S3
        s3_key = f"{user_email}/linkedin_posts.txt"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=text,
            ContentType='text/plain'
        )

        # 2. Process text
        chunks = TEXT_SPLITTER.split_text(text)
        
        # 3. Generate embeddings and prepare vectors
        vectors = []
        for chunk in chunks:
            response = bedrock.invoke_model(
                body=json.dumps({"inputText": chunk}),
                modelId="amazon.titan-embed-text-v2:0",
                accept="application/json"
            )
            embedding = json.loads(response['body'].read())['embedding']
            
            vectors.append({
                "id": f"{user_email}-{hash(chunk)}",
                "values": embedding,
                "metadata": {
                    "user_email": user_email,
                    "text": chunk,
                    "source": f"s3://{BUCKET_NAME}/{s3_key}"
                }
            })

        # 4. Store in Pinecone
        if vectors:
            print(vectors)
            pinecone_index.upsert(vectors=vectors)

    except Exception as e:
        print(f"Background processing failed: {str(e)}")
        # Implement your error handling logic here

def process_csv_and_store(user_email: str, csv_content: str):
    """Background task handling CSV processing, S3 upload and Pinecone storage"""
    try:
        # 1. Process CSV to extract text
        csv_file = io.StringIO(csv_content)
        csv_reader = csv.DictReader(csv_file)
        
        all_posts_text = ""
        for row in csv_reader:
            post_content = row.get('post_content', '')
            post_url = row.get('post_url', '')
            
            if post_content:
                all_posts_text += post_content + "\n\n"
        
        # 2. Upload raw CSV to S3
        s3_csv_key = f"{user_email}/linkedin_posts.csv"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_csv_key,
            Body=csv_content,
            ContentType='text/csv'
        )
        
        # 3. Process combined text same as before
        process_and_store(user_email, all_posts_text)

    except Exception as e:
        print(f"CSV processing failed: {str(e)}")
        # Implement your error handling logic here

@app.post("/upload")
async def upload_handler(request: UploadRequest, background_tasks: BackgroundTasks):
    """Main endpoint for LinkedIn post processing"""
    if not request.user_email or not request.linkedin_posts:
        raise HTTPException(status_code=400, detail="Missing required fields")

    background_tasks.add_task(
        process_and_store,
        request.user_email,
        request.linkedin_posts
    )

    return {
        "message": "Processing started",
        "details": {
            "user_email": request.user_email,
            "status": "background_processing"
        }
    }

@app.post("/upload-csv")
async def upload_csv_handler(request: CSVUploadRequest, background_tasks: BackgroundTasks):
    """Endpoint for LinkedIn post processing from CSV"""
    if not request.user_email or not request.linkedin_posts_csv:
        raise HTTPException(status_code=400, detail="Missing required fields")

    background_tasks.add_task(
        process_csv_and_store,
        request.user_email,
        request.linkedin_posts_csv
    )

    return {
        "message": "CSV processing started",
        "details": {
            "user_email": request.user_email,
            "status": "background_processing"
        }
    }
