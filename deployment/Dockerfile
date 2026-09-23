
# Use Python 3.12 as the base image for the deployment environment
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the deployment files into the container
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for running the application
RUN useradd -m -u 1000 user
USER user

# Set the home directory and update the path for user-installed packages
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory for the non-root user
WORKDIR $HOME/app

# Copy the deployment files and assign ownership to the non-root user
COPY --chown=user . $HOME/app

# Expose the port used by Streamlit
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
