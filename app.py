from io import BytesIO
from PyPDF2 import PdfReader
from flask import Flask, request, jsonify,session
from flask_swagger_ui import get_swaggerui_blueprint
from pymongo import MongoClient,DESCENDING, ASCENDING
from flask_cors import CORS
import os
from paddleocr import PaddleOCR
import tempfile
from pdf2image import convert_from_path
import numpy as np 
from dotenv import load_dotenv
import secrets
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.llms import OpenAI
import boto3
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import urllib.parse
from datetime import datetime
from time import time as timer
import threading
import pytz
import cv2
import numpy as np
import pre_post_processing as processing
from pdf2image import convert_from_path
from openvino1 import (image_preprocess, det_compiled_model,det_output_layer,post_processing_detection,
                        prep_for_rec,batch_text_box,rec_compiled_model,rec_output_layer)



load_dotenv()  # take environment variables from .env

app = Flask(__name__)
cors = CORS(app)
app.secret_key = "123"

os.environ["OPENAI_API_KEY"] =os.getenv("OPEN_API_KEY")

# BUCKET_NAME = os.getenv('S3_BUCKT_NAME')
# REGION = os.getenv('S3_REGION')
# IDENTITY_POOL_ID = os.getenv('COGNITO_IDENTITY_POOL_ID')


# cognito_client = boto3.client('cognito-identity',region_name=REGION)

# response = cognito_client.get_id(
#     IdentityPoolId=IDENTITY_POOL_ID
# )

# identity_id = response['IdentityId']

# credentials = cognito_client.get_credentials_for_identity(
#     IdentityId=identity_id
# )

# access_key_id = credentials['Credentials']['AccessKeyId']
# secret_access_key = credentials['Credentials']['SecretKey']
# session_token = credentials['Credentials']['SessionToken']


# s3 = boto3.client('s3',
#                     aws_access_key_id=access_key_id,
#                     aws_secret_access_key= secret_access_key,
#                      )
# s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

# response = s3.list_objects_v2(Bucket=BUCKET_NAME)

# for obj in response['Contents']:
#     print(obj['Key'])

s3 = boto3.client('s3',
                    aws_access_key_id=os.getenv("aws_access_key_id"),
                    aws_secret_access_key= os.getenv("aws_secret_access_key"),
                     )


BUCKET_NAME='scanflowpdf'



ocr = PaddleOCR(use_angle_cls=True,use_gpu=False, lang='en')
#enable_mkldnn=True

embeddings = OpenAIEmbeddings()

chat_history = []

mongo_uri = os.getenv("mongo_uri")
client = MongoClient(mongo_uri)
db = client['qa']
files=db['files']
history = db['conversations']
users = db ['users']


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    user_name = data['username']
    password = data['password']

    if user_name and password:
        existing_user = db.users.find_one({'username': user_name})
        if existing_user:
            return jsonify({'message': 'Username already exists'}), 400

        db.users.insert_one({'username': user_name, 'password': password})

        return jsonify({'message': 'User registered successfully 1', 'username': user_name}), 200
    else:
        return jsonify({'message': 'Missing username or password'}), 400
    
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user_name = data['username']
    password = data['password']

    if user_name and password:
        user = users.find_one({'username': user_name, 'password': password})
        if user:
            session['user_id'] = user['username']
            return jsonify({'message': 'Login successful','username': user_name}), 200
        else:
            return jsonify({'message': 'Invalid username or password'}), 401
    else:
        return jsonify({'message': 'Missing username or password'}), 400
    
@app.route('/files', methods=['POST'])
def files():
    try:
        data = request.get_json()
        user_id = data['username']
        # user_id = session.get('user_id')
        files = db.files.find({'user_id': user_id}).sort([('uploaded_datetime', -1),('last_chat', -1)]) 
        file_list = []
        for file in files:
            file_list.append({'file_name': file['file_name'], 'file_id': file['file_id'],
                              'uploaded_datetime':file['uploaded_datetime'],'last_chat':file['last_chat']})

        if file_list:
            return jsonify({'files': file_list}), 200
        else:
            return jsonify({'message': 'No files available'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@app.route("/upload_file_OCR", methods=["POST"])
def process_request_ocr():
    global docsearch, uploaded_pdf_data
    message=[]
    try:
        pdf_file = request.files["pdf"]
        user_id = request.form.get('username')
        tz = pytz.timezone('Asia/Kolkata')
        cur_date = datetime.now(tz).isoformat() 
        datetime_obj = datetime.strptime(cur_date, "%Y-%m-%dT%H:%M:%S.%f%z")
        cur_date = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        uploaded_file_name = pdf_file.filename
        uploaded_pdf_data = pdf_file.read()
        if not uploaded_pdf_data:
            return jsonify({'error': 'Empty file, not proceed'}),404
        existing_file = db.files.find_one({'$and': [{'user_id': user_id}, {'file_name': uploaded_file_name}]})
        if existing_file:
            return jsonify ({"message":"File already exists. Please choose a different filename"})
        else:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(uploaded_pdf_data)
                pdf_path = temp_pdf.name
            s3.upload_file(
                Bucket=BUCKET_NAME,
                Filename=temp_pdf.name,
                Key=uploaded_file_name
            )
            session['file_id'] = secrets.token_hex(16)
            # user_id = session.get('user_id')
            file_id = session['file_id']
            thread = threading.Thread(target=process_ocr, args=(pdf_path, file_id,user_id))
            thread.start()
            db.files.insert_one({'user_id':user_id,'file_id': file_id,
                                  'file_name': uploaded_file_name, 'uploaded_datetime':cur_date,
                                  'last_chat':None})
            db.history.insert_one({'user_id': user_id, 'file_id': file_id,'messages':message, 'flag':False})

            return jsonify({'message': 'File uploaded successfully',
                        'file_name': '{}'.format(uploaded_file_name), 'file_id': '{}'.format(file_id)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
def process_ocr(pdf_path, file_id,user_id):
    images = convert_from_path(pdf_path)
    os.remove(pdf_path)
    start_time = timer()
    print(start_time)
    extracted_text = extract_text_from_images(images)
    end_time = timer()
    print(end_time)
    processing_time = end_time - start_time
    print(processing_time)
    raw_text = ' '.join(extracted_text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = splitter.create_documents([raw_text])
    docsearch = FAISS.from_documents(texts, embeddings)
    docsearch.save_local(f"{file_id}")
    db.history.update_one({'user_id': user_id, 'file_id': file_id}, {'$set': {'flag': True}})

def extract_text_from_images(images):
    extracted_text = []
    for image in images:
        result = ocr.ocr(np.array(image), cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                content = line[1][0]
                content = content.replace("'", "")
                extracted_text.append(content)
    return extracted_text  


@app.route("/upload_file_Openvino", methods=["POST"])
def process_request_openvino():
    global docsearch, uploaded_pdf_data
    message=[]
    try:
        pdf_file = request.files["pdf"]
        user_id = request.form.get('username')
        tz = pytz.timezone('Asia/Kolkata')
        cur_date = datetime.now(tz).isoformat() 
        datetime_obj = datetime.strptime(cur_date, "%Y-%m-%dT%H:%M:%S.%f%z")
        cur_date = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        uploaded_file_name = pdf_file.filename
        uploaded_pdf_data = pdf_file.read()
        if not uploaded_pdf_data:
            return jsonify({'error': 'Empty file, not proceed'}),404
        existing_file = db.files.find_one({'$and': [{'user_id': user_id}, {'file_name': uploaded_file_name}]})
        if existing_file:
            return jsonify ({"message":"File already exists. Please choose a different filename"})
        else:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(uploaded_pdf_data)
                pdf_path = temp_pdf.name
            s3.upload_file(
                Bucket=BUCKET_NAME,
                Filename=temp_pdf.name,
                Key=uploaded_file_name
            )
            session['file_id'] = secrets.token_hex(16)
            # user_id = session.get('user_id')
            file_id = session['file_id']
            thread = threading.Thread(target=process_openvino, args=(pdf_path, file_id,user_id))
            thread.start()
            db.files.insert_one({'user_id':user_id,'file_id': file_id,
                                  'file_name': uploaded_file_name, 'uploaded_datetime':cur_date,
                                  'last_chat':None})
            db.history.insert_one({'user_id': user_id, 'file_id': file_id,'messages':message, 'flag':False})

            return jsonify({'message': 'File uploaded successfully',
                        'file_name': '{}'.format(uploaded_file_name), 'file_id': '{}'.format(file_id)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
def process_openvino(pdf_path, file_id,user_id):
    extracted_text = []
    images = convert_from_path(pdf_path)
    start_time = timer()
    print(start_time)
    for image in images:
        image_array = np.array(image)
        text=run_paddle_ocr(image_array)
        extracted_text.extend(text)
    print(extracted_text)
    end_time = timer()
    print(end_time)
    processing_time = end_time - start_time
    print(processing_time)
    raw_text = ' '.join(extracted_text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = splitter.create_documents([raw_text])
    docsearch = FAISS.from_documents(texts, embeddings)
    docsearch.save_local(f"{file_id}")
    db.history.update_one({'user_id': user_id, 'file_id': file_id}, {'$set': {'flag': True}})


def run_paddle_ocr(frame):
    try:
        if frame is None:
            print("Source ended")
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(src=frame, dsize=None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
        test_image = image_preprocess(frame, 960)

        det_results = det_compiled_model([test_image])[det_output_layer]

        dt_boxes = post_processing_detection(frame, det_results)

        dt_boxes = processing.sorted_boxes(dt_boxes)
        batch_num = 6
        img_crop_list, img_num, indices = prep_for_rec(dt_boxes, frame)

        rec_res = [['', 0.0]] * img_num
        txts = []

        for beg_img_no in range(0, img_num, batch_num):

            norm_img_batch = batch_text_box(
                img_crop_list, img_num, indices, beg_img_no, batch_num)

            rec_results = rec_compiled_model([norm_img_batch])[rec_output_layer]

            postprocess_op = processing.build_post_process(processing.ch_postprocess_params)
            rec_result = postprocess_op(rec_results)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            if rec_res:
                txts = [rec_res[i][0] for i in range(len(rec_res))]
        return txts
    except KeyboardInterrupt:
        print("Interrupted")
    except RuntimeError as e:
        print(e) 


# @app.route("/upload_file", methods=["POST"])
# def process_request():
#     global docsearch, uploaded_pdf_data
#     try:
#         pdf_file = request.files["pdf"]
#         user_id = request.form.get('username')
#         tz = pytz.timezone('Asia/Kolkata')
#         cur_date = datetime.now(tz).isoformat() 
#         datetime_obj = datetime.strptime(cur_date, "%Y-%m-%dT%H:%M:%S.%f%z")
#         cur_date = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
#         uploaded_file_name = pdf_file.filename
#         uploaded_pdf_data = pdf_file.read()
#         if not uploaded_pdf_data:
#             return jsonify({'error': 'Empty file'}),404
#         existing_file = db.files.find_one({'$and': [{'user_id': user_id}, {'file_name': uploaded_file_name}]})
#         if existing_file:
#             return jsonify ({"message":"File already exists. Please choose a different filename"}),409
#         else:
#             with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
#                 temp_pdf.write(uploaded_pdf_data)
#                 pdf_path = temp_pdf.name
#             s3.upload_file(
#                 Bucket=BUCKET_NAME,
#                 Filename=temp_pdf.name,
#                 Key=uploaded_file_name
#             )
#             # s3_file_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{urllib.parse.quote(uploaded_file_name)}"

#             session['file_id'] = secrets.token_hex(16)
#             # user_id = session.get('user_id')
#             file_id = session['file_id']
#             reader = PdfReader(BytesIO(uploaded_pdf_data))
#             raw_text = ""
#             for page in reader.pages:
#                 text = page.extract_text()
#                 if text:
#                     raw_text += text 
#             print(raw_text)
#             splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#             texts = splitter.create_documents([raw_text])
#             docsearch = FAISS.from_documents(texts, embeddings)
#             docsearch.save_local(f"{file_id}")
#             db.files.insert_one({'user_id':user_id,'file_id': file_id,
#                                   'file_name': uploaded_file_name, 'uploaded_datetime':cur_date,
#                                   'last_chat':None})


#             return jsonify({'message': 'File uploaded successfully',
#                         'file_name': '{}'.format(uploaded_file_name), 'file_id': '{}'.format(file_id)}), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400
        

@app.route("/question", methods=["POST"])  
def question_request():
    global chat_history
    try:
        data = request.get_json()
        user_id = data['username']
        file_id = data['file_id']
        question = data['question']
        tz = pytz.timezone('Asia/Kolkata')
        cur_date = datetime.now(tz).isoformat() 
        datetime_obj = datetime.strptime(cur_date, "%Y-%m-%dT%H:%M:%S.%f%z")
        cur_date = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

        result = db.files.find_one({'file_id': file_id})
        file_name = result['file_name']
        new_db = FAISS.load_local(f"{file_id}", embeddings=embeddings)
        if not new_db:
            return jsonify({'error': 'File not found'}),404

        docs = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        qn_chain = ConversationalRetrievalChain.from_llm(OpenAI(), docs)
        result = qn_chain({"question": question, "chat_history": chat_history})
        answer = result["answer"]
        chat_history.append((question, answer))

            # user_id = session.get('user_id')
        if user_id:
            store_messages(user_id, file_id, [{'file_name':file_name, 'question':question, 'answer':answer}])
        db.files.update_one({'user_id':user_id, 'file_id':file_id},{'$set':{'last_chat':cur_date}})

        return jsonify({
                'question': question,
                'answer': answer,
                'chat_history': retrieve_conversation(user_id,file_id)
        }),200

    except Exception as e:
        return jsonify({'error': str(e)}),400
    
@app.route("/history", methods=["POST"])
def history():
    try:
        data = request.get_json()
        user_id = data['username']
        file_id = data['file_id']
        # user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID not found'})

        conversations = retrieve_conversation(user_id,file_id)
        return jsonify({'messages':conversations['messages'], 'flag':conversations['flag']}),200

    except Exception as e:
        return jsonify({'error': str(e)}),400
    

def store_messages(user_id, file_id, messages):
    conversation = db.history.find_one({'user_id': user_id, 'file_id': file_id})
    conversation['messages'].extend(messages)
    db.history.update_one({'user_id': user_id, 'file_id': file_id}, {'$set': {'messages': conversation['messages']}})

def retrieve_conversation(user_id, file_id):
    conversation = db.history.find_one({'$and': [{'user_id': user_id}, {'file_id': file_id}]})
    return {'messages':conversation['messages'], 'flag':conversation['flag']}

      
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Scanflow"}
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

  