import socket
import pickle
import numpy

from sklearn.metrics import accuracy_score
from src.method.hydra.custom_training import HYDRA_Training

model = HYDRA_Training()
model.init_model()

def recv(soc, buffer_size=1024, recv_timeout=10):
    received_data = b""
    while str(received_data)[-2] != '.':
        try:
            soc.settimeout(recv_timeout)
            received_data += soc.recv(buffer_size)
        except socket.timeout:
            print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds. There may be an error or the model may be trained successfully.".format(recv_timeout=recv_timeout))
            return None, 0
        except BaseException as e:
            print("An error occurred while receiving data from the server {msg}.".format(msg=e))
            return None, 0

    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
        return None, 0

    return received_data, 1

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

try:
    soc.connect(("localhost", 3000))
    print("Successful Connection to the Server.\n")
except BaseException as e:
    print("Error Connecting to the Server: {msg}".format(msg=e))
    soc.close()
    print("Socket Closed.")

subject = "echo"
GANN_instance = None

while True:
    data = {"subject": subject, "data": GANN_instance}
    data_byte = pickle.dumps(data)
    
    print("Sending the Model to the Server.\n")
    print(data)
    soc.sendall(data_byte)
    
    print("Receiving Reply from the Server.")
    received_data, status = recv(soc=soc, 
                                 buffer_size=1024, 
                                 recv_timeout=10)
    if status == 0:
        print("Nothing Received from the Server.")
        break
    else:
        print(received_data, end="\n\n")

    subject = received_data["subject"]
    if subject == "model":
        GANN_instance = received_data["data"]
    elif subject == "done":
        print("The server said the model is trained successfully and no need for further updates its parameters.")
        break
    else:
        print("Unrecognized message type.")
        break

    model.load_weights(weights=GANN_instance)
    weights = model.train()

    subject = "model"
    print("Sending the Updated Model to the Server.\n")

# predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[best_sol_idx], data_inputs=data_inputs)

soc.close()
print("Socket Closed.\n")