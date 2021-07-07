import matplotlib
from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')

def plotting(losses_g,losses_d):

    plt.figure(1)
    plt.clf()
    plt.title("Traning")
    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.plot(losses_g, label='Generator loss')
    plt.plot(losses_d, label='Discriminator Loss')
    plt.legend()
    plt.pause(0.001)