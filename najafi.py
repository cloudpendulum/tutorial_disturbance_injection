import numpy as np

#from termios import TIOCSWINSZ
from scipy.spatial.transform import Rotation as R
from scipy import linalg
from scipy.special import gamma, factorial
import numpy as np
#from pydrake.all import (MathematicalProgram, Solve, Variables, Jacobian)
#from pydrake.symbolic import TaylorExpand, Evaluate
#from pydrake.all import Variable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

def get_ellipse_params(rho,M):
    """
    Returns ellipse params (excl center point)
    """

    #eigenvalue decomposition to get the axes
    w,v=np.linalg.eigh(M/rho) 

    try:
        #let the smaller eigenvalue define the width (major axis*2!)
        width = 2/float(np.sqrt(w[0]))
        height = 2/float(np.sqrt(w[1]))
        
        #the angle of the ellipse is defined by the eigenvector assigned to the smallest eigenvalue (because this defines the major axis (width of the ellipse))
        angle = np.rad2deg(np.arctan2(v[:,0][1],v[:,0][0]))

    except:
        print("paramters do not represent an ellipse.")

    return width,height,angle

def get_ellipse_patch(px,py,rho,M,alpha_val=1,linec="red",facec="none",linest="solid"):
    """
    return an ellipse patch
    """
    w,h,a = get_ellipse_params(rho,M)
    print("w: ", w)
    print("h: ", h)
    print("a: ", a)
    return patches.Ellipse((px,py), w, h, angle=a, alpha=alpha_val,ec=linec,facecolor=facec,linestyle=linest)

def plot_ellipse(px,py,rho, M, points, save_to=None, show=True):
    p=get_ellipse_patch(px,py,rho,M)
    
    fig, ax = plt.subplots()
    plt.scatter(np.asarray(points).T[0], np.asarray(points).T[1], s=1)
    ax.add_patch(p)
    l=np.max([p.width,p.height])

    ax.set_xlim(px-l/2,px+l/2)
    ax.set_ylim(py-l/2,py+l/2)

    ax.grid(True)

    if not (save_to is None):
        plt.savefig(save_to)
    if show:
        plt.show()

def direct_sphere(d,r_i=0,r_o=1):
    """Direct Sampling from the d Ball based on Krauth, Werner. Statistical Mechanics: Algorithms and Computations. Oxford Master Series in Physics 13. Oxford: Oxford University Press, 2006. page 42

    Parameters
    ----------
    d : int
        dimension of the ball
    r_i : int, optional
        inner radius, by default 0
    r_o : int, optional
        outer radius, by default 1

    Returns
    -------
    np.array
        random vector directly sampled from the solid d Ball
    """
    # vector of univariate gaussians:
    rand=np.random.normal(size=d)
    # get its euclidean distance:
    dist=np.linalg.norm(rand,ord=2)
    # divide by norm
    normed=rand/dist
    
    # sample the radius uniformly from 0 to 1 
    rad=np.random.uniform(r_i,r_o**d)**(1/d)
    # the r**d part was not there in the original implementation.
    # I added it in order to be able to change the radius of the sphere
    # multiply with vect and return
    return normed*rad

def sample_from_ellipsoid(M,rho,r_i=0,r_o=1):
    """sample directly from the ellipsoid defined by xT M x.

    Parameters
    ----------
    M : np.array
        Matrix M such that xT M x leq rho defines the hyperellipsoid to sample from
    rho : float
        rho such that xT M x leq rho defines the hyperellipsoid to sample from
    r_i : int, optional
        inner radius, by default 0
    r_o : int, optional
        outer radius, by default 1

    Returns
    -------
    np.array
        random vector from within the hyperellipsoid
    """
    lamb,eigV=np.linalg.eigh(M/rho) 
    d=len(M)
    xy=direct_sphere(d,r_i=r_i,r_o=r_o) #sample from outer shells
    T=np.linalg.inv(np.dot(np.diag(np.sqrt(lamb)),eigV.T)) #transform sphere to ellipsoid (refer to e.g. boyd lectures on linear algebra)
    return np.dot(T,xy.T).T

def quad_form(M,x):
    """
    Helper function to compute quadratic forms such as x^TMx
    """
    return np.dot(x,np.dot(M,x))


def vol_ellipsoid(rho,M):
    """
    Calculate the Volume of a Hyperellipsoid
    Volume of the Hyperllipsoid according to https://math.stackexchange.com/questions/332391/volume-of-hyperellipsoid/332434
    Intuition: https://textbooks.math.gatech.edu/ila/determinants-volumes.html
    Volume of n-Ball https://en.wikipedia.org/wiki/Volume_of_an_n-ball
    """
    
    # For a given hyperellipsoid, find the transformation that when applied to the n Ball yields the hyperellipsoid
    lamb,eigV=np.linalg.eigh(M/rho) 
    A=np.dot(np.diag(np.sqrt(lamb)),eigV.T) #transform ellipsoid to sphere
    detA=np.linalg.det(A)
    
    # Volume of n Ball (d dimensions)
    d=M.shape[0] # dimension 
    volC=(np.pi**(d/2))/(gamma((d/2)+1))

    # Volume of Ellipse
    volE=volC/detA

    return volE

def sample_from_ellipsoid(M,rho,r_i=0,r_o=1):
    """sample directly from the ellipsoid defined by xT M x.

    Parameters
    ----------
    M : np.array
        Matrix M such that xT M x leq rho defines the hyperellipsoid to sample from
    rho : float
        rho such that xT M x leq rho defines the hyperellipsoid to sample from
    r_i : int, optional
        inner radius, by default 0
    r_o : int, optional
        outer radius, by default 1

    Returns
    -------
    np.array
        random vector from within the hyperellipsoid
    """
    lamb,eigV=np.linalg.eigh(M/rho) 
    d=len(M)
    xy=direct_sphere(d,r_i=r_i,r_o=r_o) #sample from outer shells
    T=np.linalg.inv(np.dot(np.diag(np.sqrt(lamb)),eigV.T)) #transform sphere to ellipsoid (refer to e.g. boyd lectures on linear algebra)
    return np.dot(T,xy.T).T

def najafi_based_sampling(
    plant, controller, n=10000, rho0=100, M=None, x_star=np.array([np.pi, 0])
):
    """Estimate the RoA for the closed loop dynamics using the method introduced in Najafi, E., Babuška, R. & Lopes, G.A.D. A fast sampling method for estimating the domain of attraction. Nonlinear Dyn 86, 823–834 (2016). https://doi.org/10.1007/s11071-016-2926-7

    Parameters
    ----------
    plant : simple_pendulum.model.pendulum_plant
        configured pendulum plant object
    controller : simple_pendulum.controllers.lqr.lqr_controller
        configured lqr controller object
    n : int, optional
        number of samples, by default 100000
    rho0 : int, optional
        initial estimate of rho, by default 10
    M : np.array, optional
        M, such that x_barT M x_bar is the Lyapunov fct. by default None, and controller.S is used
    x_star : np.array, optional
        nominal position (fixed point of the nonlinear dynamics)

    Returns
    -------
    rho : float
        estimated value of rho
    M : np.array
        M
    points: list containing all the points that were tested
    """

    rho = rho0

    points = []
    
    if M is None:
        M = np.array(controller.S)
    else:
        pass

    for i in range(n):
        # sample initial state from sublevel set
        # check if it fullfills Lyapunov conditions
        x_bar = sample_from_ellipsoid(M, rho)
        x = x_star + x_bar

        tau = controller.get_control_output([x[0], x[1]])

        xdot = plant.rhs(0, x, tau)

        V = x_bar.T @ M @ x_bar 

        Vdot = 2 * np.dot(x_bar, np.dot(M, xdot))

        if V > rho:
            print("something is fishy")
        # V < rho is true trivially, because we sample from the ellipsoid
        if Vdot > 0.0:  # if one of the lyapunov conditions is not satisfied
            rho = V

        points.append(x)
    
    return rho, M, points