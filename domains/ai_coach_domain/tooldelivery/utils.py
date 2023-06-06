

def read_sample(file_name):
    traj = []
    latents = []
    with open(file_name, newline='') as txtfile:
        lines = txtfile.readlines()
        latents = []
        for elem in lines[1].rstrip().split(", "):
            if elem.isdigit():
                latents.append(int(elem))
            elif elem == "None":
                latents.append(None)

        for i_r in range(3, len(lines)):
            line = lines[i_r]
            traj.append(
                tuple([int(elem) for elem in line.rstrip().split(", ")]))
    return traj, tuple(latents)