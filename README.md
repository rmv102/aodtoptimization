# A guide to Installing NVIDIA Omniverse Aerial DT (Simplified with solutions to common problems)

# 1. The requirements that worked for my system but does not say so in the official documentation:

OS: Ubuntu 22.04 LTS
Hardware: 

  GPU: 2x NVIDIA A6000 Ada 
  VCPUs: 12
  RAM: 48
  Storage: 150 GB

*If you are using 2 GPus in 1 machine, you need to change the docker-compose.yml file so the application doesn't only run on one GPU*


# 2. Before you can install the application, you need to join the 6G developer program for NVIDIA Omniverse: https://developer.nvidia.com/6g-program
Next, head on over to the the NVIDIA website, create an account and get an API key from the NGC Catalog: https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html

NOTE: DO NOT MAKE A "Personal Key", YOU NEED TO MAKE AN OFFICIAL API KEY, OTHERWISE IT WILL NOT WORK


One you have the information above, here are the following commands you need to perform on your virtual machine:

sudo apt-get install -y jq unzip 

export NGC_CLI_API_KEY=<NGC_CLI_API_KEY>
AUTH_URL="https://authn.nvidia.com/token?service=ngc&scope=group/ngc:esee5uzbruax&group/ngc:esee5uzbruax/"

TOKEN=$(curl -s -u "\$oauthtoken":"$NGC_CLI_API_KEY" -H "Accept:application/json" "$AUTH_URL" | jq -r '.token')

versionTag="1.1.0"
downloadedZip="$HOME/aodt_bundle.zip"

curl -L "https://api.ngc.nvidia.com/v2/org/esee5uzbruax/resources/aodt-installer/versions/$versionTag/files/aodt_bundle.zip" -H "Authorization: Bearer$TOKEN" -H "Content-Type: application/json" -o $downloadedZip

ALternatively, you can use your API Key to download the zip files straight from the NGC catalog. It will produce the same aodt_bundle.zip.

# 3. To actually install and run it, do:

./aodt_bundle/install.sh localhost $NGC_CLI_API_KEY



The output should say something about a vnc server running.

NOTE: DO NOT TRY INSTALLING IT AGAIN, IT DOESN't RUN IF THERE ARE TWO INSTALLATIONS. Also, another problem comes up when you add quotations when you are installing from a website using curl. The symbol " is not the same across all operating systems, and this installation doesn't install the whole thing, but only the webpage. You will know the installation is working if a 8gb file is being downloaded.

Next, you need to get a VNC client to log into the system. If you are using a VM, tunnel an external port to internal port 5901. When prompted with a password, enter "nvidia"

# 4. Opening the launcher and starting the backend
Once the launcher starts up, you will see NVIDIA Aerial DT in the library tab. Before you launch it, you must go to the "backend_bundle" folder and Update the "docker-compose.yml" file. There should be an array named "gpus" that says ['0']. In order for both the GPUs to be used (becauset he whole application is only running on one gpu), you have to change that '0' in the array to '1'. So it should look something like gpus: ['1'];

Then, you can simply run "docker-compose up" to start up the backend. 

When you launch the application, the username is "omniverse" and password is "aerial_123456" 

NOTE: IN THE ACTUAL CONFIGURATION OF THE NVIDIA AERIAL CONFIGURATION TAB, YOU CANNOT PUT "localhost" FOR YOUR BACKEND INFORMATION, YOU MUST PUT THE LOCAL IP ADDRESS OF YOUR SYSTEM. For me, I just replaced localhost with 192.168.75.133 in order to connect the "worker".












