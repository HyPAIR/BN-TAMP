from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import subprocess
import socket
import os

#communicating with coppelia sim and using it different modules
class Connection:
    #coppeliaExeDir = "~/coppelia/CoppeliaSim_Edu_V4_6_0_rev16_Ubuntu22_04/./coppeliaSim.sh" 
    #parentDir = os.path.abspath(os.path.join(os.getcwd(),os.pardir))
    #coppeliaEnvDir = parentDir+"/Env/Sensory_data_collection.ttt"
    #ip = 'localhost'
    #sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    def __init__(self):
        #connect to Coppelia sim
        #establishing server
        #open = Connection.sock.connect_ex((Connection.ip,port))
        #if the port is closed create server
        #if(open!=0): 
         #  os.system(Connection.coppeliaExeDir+" -GzmqRemoteApi.rpcPort="+str(port)+" "+
            #          Connection.coppeliaEnvDir)
        #else client communicate directly with the existing one
        self.__client = RemoteAPIClient()
        self.sim = self.__client.require('sim') #simulation
        self.simIK = self.__client.require('simIK') #inverse kinematics
        self.simOMPL = self.__client.require('simOMPL') #path planning