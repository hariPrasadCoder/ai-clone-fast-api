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
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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
pinecone_index = pc.Index("user-posts-with-url")

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
            pinecone_index.upsert(vectors=vectors)

    except Exception as e:
        print(f"Background processing failed: {str(e)}")
        # Implement your error handling logic here

def to_date(relative: str, anchor=pd.Timestamp("today").normalize()):
    """Convert strings like '3mo', '2w', '1d', '1yr' into a concrete date."""
    if relative.endswith("d"):
        return anchor - timedelta(days=int(relative[:-1]))
    if relative.endswith("w"):
        return anchor - timedelta(weeks=int(relative[:-1]))
    if relative.endswith("mo"):
        return anchor - relativedelta(months=int(relative[:-2]))
    if relative.endswith("yr"):
        return anchor - relativedelta(years=int(relative[:-2]))
    raise ValueError(f"Unrecognised offset: {relative}")

def process_csv_and_store(user_email: str, csv_content: str):
    """Background task handling CSV processing, S3 upload and Pinecone storage"""
    try:
        # 1. Process CSV to extract text
        df = pd.read_csv(io.StringIO(csv_content))
        df["post_date"] = df["post_timestamp"].apply(to_date).astype(str)
        
        # 2. Upload raw CSV to S3
        s3_csv_key = f"{user_email}/linkedin_posts.csv"
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_csv_key,
            Body=csv_content,
            ContentType='text/csv'
        )
        
        vectors = []
        for i, row in df.iterrows():
            response = bedrock.invoke_model(
                body=json.dumps({"inputText": row['post_content']}),
                modelId="amazon.titan-embed-text-v2:0",
                accept="application/json"
            )
            embedding = json.loads(response['body'].read())['embedding']
            
            vectors.append({
                "id": f"{user_email}-{hash(row['post_content'])}",
                "values": embedding,
                "metadata": {
                    "user_email": user_email,
                    "text": row['post_content'],
                    "source": f"s3://{BUCKET_NAME}/{s3_csv_key}",
                    "post_url": row['post_url'],
                    "post_date": row['post_date']
                }
            })

        # 4. Store in Pinecone
        if vectors:
            pinecone_index.upsert(vectors=vectors)

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
