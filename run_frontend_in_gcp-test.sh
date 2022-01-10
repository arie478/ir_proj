# 0. Install Cloud SDK on your local machine or using Could Shell
# check that you have a proper active account listed
gcloud auth list 
# check that the right project and zone are active
gcloud config list
# if not set them
# gcloud config set project assignment-3-gcp-334112
# gcloud config set compute/zone us-central1-c

# 1. Set up public IP
gcloud compute addresses create assignment-3-gcp-334112-ip --project=assignment-3-gcp-334112 --region=us-central1
gcloud compute addresses list
# note the IP address printed above, that's your extrenal IP address.
# Enter it here: 
# INSTANCE_IP="35.194.18.78"

# 2. Create Firewall rule to allow traffic to port 8080 on the instance
gcloud compute firewall-rules create default-allow-http-8080 --allow tcp:8080 --source-ranges 0.0.0.0/0 --target-tags http-server

# 3. Create the instance. Change to a larger instance (larger than e2-micro) as needed.
gcloud compute instances create instance-1 --zone=us-central1-c --machine-type=e2-standard-4 --network-interface=address=35.194.18.78,network-tier=PREMIUM,subnet=default --metadata-from-file startup-script=startup_script_gcp.sh --scopes=https://www.googleapis.com/auth/cloud-platform --tags=http-server --boot-disk-size 20GB
#gcloud compute instances create instance-1 --zone=us-central1-c --machine-type=e2-micro --network-interface=address=34.123.197.120,network-tier=PREMIUM,subnet=default --metadata-from-file startup-script=startup_script_gcp.sh --scopes=https://www.googleapis.com/auth/cloud-platform --tags=http-server
# monitor instance creation log using this command. When done (4-5 minutes) terminate using Ctrl+C
gcloud compute instances tail-serial-port-output instance-1 --zone us-central1-c

# 4. Secure copy your app to the VM
gcloud compute scp E:/index/GCP_RUN/search_frontend.py ariekat@instance-1:/home/ariekat

gcloud compute scp E:/index/GCP_RUN/gcp-key-334112-88fd1da949d9.json ariekat@instance-1:/home/ariekat
gcloud compute scp E:/index/GCP_RUN/inverted_index_gcp_with_dl.py ariekat@instance-1:/home/ariekat

# 5. SSH to your VM and start the app
gcloud compute ssh ariekat@instance-1
python3 search_frontend.py

################################################################################
# Clean up commands to undo the above set up and avoid unnecessary charges
gcloud compute instances delete -q instance-1
# make sure there are no lingering instances
gcloud compute instances list
# delete firewall rule
gcloud compute firewall-rules delete -q default-allow-http-8080
# delete external addresses
gcloud compute addresses delete -q assignment-3-gcp-334112-ip --region us-central1