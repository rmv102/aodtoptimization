# A Roadmap to Develop DT Leveraging NVIDIA Omniverse (Simplified with solutions to common problems)

## 1. System requirements:

OS: Ubuntu 22.04 LTS
Hardware: 

  GPU: 2x NVIDIA A6000 Ada 
  VCPUs: 12
  RAM: 48
  Storage: 150 GB

*If you are using 2 GPus in 1 machine, you need to change the docker-compose.yml file so the application doesn't only run on one GPU*


## 2. Before you can install the application, you need to join the 6G developer program for NVIDIA Omniverse: https://developer.nvidia.com/6g-program
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

## 3. To actually install and run it, do:

./aodt_bundle/install.sh localhost $NGC_CLI_API_KEY



The output should say something about a vnc server running.

NOTE: DO NOT TRY INSTALLING IT AGAIN, IT DOESN't RUN IF THERE ARE TWO INSTALLATIONS. Also, another problem comes up when you add quotations when you are installing from a website using curl. The symbol " is not the same across all operating systems, and this installation doesn't install the whole thing, but only the webpage. You will know the installation is working if a 8gb file is being downloaded.

Next, you need to get a VNC client to log into the system. If you are using a VM, tunnel an external port to internal port 5901. When prompted with a password, enter "nvidia"

## 4. Opening the launcher and starting the backend
Once the launcher starts up, you will see NVIDIA Aerial DT in the library tab. Before you launch it, you must go to the "backend_bundle" folder and Update the "docker-compose.yml" file. There should be an array named "gpus" that says ['0']. In order for both the GPUs to be used (becauset he whole application is only running on one gpu), you have to change that '0' in the array to '1'. So it should look something like gpus: ['1'];

Then, you can simply run "docker-compose up" to start up the backend. 

When you launch the application, the username is "omniverse" and password is "aerial_123456" 

NOTE: IN THE ACTUAL CONFIGURATION OF THE NVIDIA AERIAL CONFIGURATION TAB, YOU CANNOT PUT "localhost" FOR YOUR BACKEND INFORMATION, YOU MUST PUT THE LOCAL IP ADDRESS OF YOUR SYSTEM. For me, I just replaced localhost with 192.168.75.133 in order to connect the "worker".


## 5. Obtain data and coverage maps from AODT

Input the python script into your clickhouse database buy putting the script into your folder. This script generates a csv file with RU coordinates and creates a heatmap out of the signal strength values. Then use an online heatmap generator that accepts csvs (e.g. http://www.heatmapper.ca/geocoordinate/). This tool helps good visuals like this:


![image](https://github.com/user-attachments/assets/38e1231a-bd23-4609-97d3-3d42df15d637)

![image](https://github.com/user-attachments/assets/58ad7342-1d78-47e3-b11b-a1e8057c3715)

![image](https://github.com/user-attachments/assets/840cd2f4-9790-4920-ad3d-a31972423755)

3 Base Stations:
![image](https://github.com/user-attachments/assets/2b83c732-c665-4f77-820c-a0d5ba490d7b)





## 6. Provided example scripts

There are two scripts in this github.

### Data Generation 
One of them is **baseStationCoordinates.py**. This script is designed to make a csv file with a base station configuration. You can then visit sites like http://www.heatmapper.ca/geocoordinate/ to generate the respective visualized heatmaps.

### Optimizing Infrastracture Planning 
The other script, **base_station_brute_force_script.py**, this script picks buildings in urban environment (e.g. Raleigh City, NC) and based on the number of deployed 5G base stations, the script automates the process of placing base stations on buildings (buildings are found in .USD maps by a certain given height). Then, for every case, it adds up all the scores of the signal strength and returns the highest. For example, and output might look like this:




![image](https://github.com/user-attachments/assets/92963036-80c1-474e-bb51-0b0580e5f9c9)



## 7. Demos (Cluster based optimization)

Here are a bunch of trial runs that were performed using the two scripts mentioned. These are all triangle shapes divided into 500x500 meter clusters.


### Single Cluster Optimization
Here is a singular cluster block shape with 3 base stations inside. This is the heatmap of the optimal position of the base stations, and the radio icons indicate the approximate location of these base stations on the actual map. The variance in the shape and structure has to do with many factors, such as terrain elevation, surrounding buildings, interference sources, etc.

![image](https://github.com/user-attachments/assets/691b8dcb-28fc-4e59-8708-89e933efd1ba)



### Two Clusters in a Row
In this setup, the optimization script is applied to two adjacent clusters, each measuring approximately 500x500 meters. Each cluster contains three base stations that are strategically placed to ensure seamless coverage across both areas while minimizing overlapping interference.

This scenario demonstrates how the optimization process scales when multiple clusters are involved, adapting dynamically to maintain signal strength and efficiency across a larger area.

![image](https://github.com/user-attachments/assets/5f971515-159e-4360-a501-d28f384a9640)


### Three Cluster Blocks in a Row

Expanding further, this example illustrates three consecutive cluster blocks, each optimized for three base stations. As more clusters are added, the complexity of the optimization process increases, requiring careful balancing of coverage, signal strength, and interference mitigation across all three areas.

This simulation provides insights into how base stations can be arranged in larger network deployments, ensuring continuous and reliable connectivity across extended regions.

![image](https://github.com/user-attachments/assets/4c8ba79e-7935-40f5-af42-a18fbb8b2a30)


### Close-Up View of a Single Cluster
Due to the wide area covered in previous images, resolution limitations make it difficult to capture finer details. The image below provides a close-up view of a single cluster, showcasing the more precise placement of base stations and the intricate heatmap patterns that result from the optimization process.

This close-up highlights how signal distribution is affected by environmental factors at a granular level, further refining the accuracy of base station positioning.

![image](https://github.com/user-attachments/assets/79d0016c-a2f7-4898-a3f0-49eea525c0a3)

### Stacked Cluster Configurations

Finally, more complex configurations are possible beyond linear cluster arrangements. The example below demonstrates a stacked cluster approach, where base station clusters are positioned in a layered structure rather than in a single row.
![image](https://github.com/user-attachments/assets/fbf931a0-b370-4392-beaa-279a93b4c3b5)




## Contact:
Maulik Verma

Yuchen Liu (NCSU) 

## Acknowledgement:
NVIDIA Omniverse Platform 

NAIRR Pilot Demonstration Project




