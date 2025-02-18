FROM public.ecr.aws/lambda/python:3.9

# 1. Install system libraries required by OpenCV
RUN yum -y update && yum -y install mesa-libGL mesa-libGLES

# 2. Copy your function code and requirements
COPY app.py requirements.txt ./

# 3. Install Python dependencies into /var/task
RUN pip3 install --no-cache-dir -r requirements.txt --target /var/task

# 4. Set the Lambda entrypoint
CMD ["app.handler"]
