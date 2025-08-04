
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

# Expose port (if using Flask or FastAPI)
EXPOSE 8501

CMD ["streamlit", "run", "chatapp.py"]
