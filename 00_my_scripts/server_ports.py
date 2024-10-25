

# Get the API ports from the CMD_FLAGS-Multi-Server.txt file
def get_api_ports(filename, start_port=4000):
    ports = []
    with open(filename, 'r') as file:
        for line in file: 
            if '#' not in line:
                words = line.split()
                if '--api-port' in words:
                    i = words.index('--api-port')
                    if i+1 < len(words):
                        port = int(words[i+1])
                        if port >= start_port:
                            ports.append(port)
    return ports

AVAILABLE_PORTS = get_api_ports('..\CMD_FLAGS-Multi-Server.txt')

# Dictionary mapping ports to users
PORT_USERS = {port: set([]) for port in AVAILABLE_PORTS}

# Dictionary mapping ports to their status (True = in use, False = not in use)
PORT_STATUS = {port: False for port in AVAILABLE_PORTS}

# Dictionary mapping users to ports
USER_PORTS = {}

def get_least_loaded_port():
    # Get the list of ports that are not in use
    not_in_use_ports = [port for port in AVAILABLE_PORTS if not PORT_STATUS[port]]

    # If there are ports not in use, return the least loaded one
    if not_in_use_ports:
        return min(not_in_use_ports, key=lambda port: len(PORT_USERS[port]))

    # If all ports are in use, return the least loaded one
    return min(PORT_USERS, key=lambda port: len(PORT_USERS[port]))