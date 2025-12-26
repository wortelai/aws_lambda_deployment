# Start from AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Copy your code
COPY . ${LAMBDA_TASK_ROOT}/

# Install your dependencies
RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
RUN pip install pytest

# Set the CMD to your handler function (module.function)
CMD ["lambda_handler.handler"]


#################### testing################

############# convert image to string############
# base64 input.jpg > image_base64.txt

################ Create a JSON payload file#############
# echo "{\"image_b64\": \"$(cat image_base64.txt)\"}" > payload.json

############## call lambda function##################
# curl -s -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
#   -H "Content-Type: application/json" \
#   --data @payload.json \
#   | jq -r '.body | fromjson | .image_base64' \
#   | base64 --decode > output.jpg
