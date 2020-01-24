#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as py

def midpoint(v_init, theta, g=1, t_period = 5):
    """
    Outline. 
  
    This function employs midpoint routine to calculate the parabolic trajectory 
    of an object under constant force in y-axis.
  
    Parameters: 
    v_init   : Initial velocity of the object 
    theta    : Initial angle by which the obeject is launced w.r.t positive x-axis
    g        : Acceleration due to gravitation
    t_period : Time period for which the simulation is run
    
    Returns: 
    r_x: Array containning x values of the object at dt time differences
    r_y: Array containning y values of the object at dt time differences    
    """
    
    dt = 0.001
    period = t_period + dt
    length = int(period / dt)
    r_x = np.zeros(length)
    r_y = np.zeros(length)
    
    r_x[0] , r_y[0] = 0.0 , 0.0
    v_x , v_y = v_init * np.cos(theta), v_init * np.sin(theta)
    
    
    for i in range(1, length):
        r_x[i] = r_x[i-1] + dt * v_x
        r_y[i] = r_y[i-1] + dt * v_y - 0.5 *g* dt**2 
        v_y -= dt*g
        
    return r_x, r_y



def plotParabolas(v = 1,g=1, n= 24,th_fac = np.pi/12, func= midpoint):
    """
    Outline. 
  
    This function plots the trajectories for an object with given function with a constant velocity
    but for varying angles
  
    Parameters: 
    n      : Number of trajectories to be plotted 
    th_fac : Difference in launching angle for the different trajectories 
    func   : Routine for which the trajectory is to be plotted
    
    Output: 
    It generates plots for trajectories with a constant velocity but different angles    
    
    """
    for i in range(n):
        theta = th_fac *i 
        r_x, r_y = func(v,theta,g,5)
        py.plot(r_x,r_y,'k-')

        
def envParabola(v=1, g=1, x=4 ,n=50, show = True):
    """
    Outline. 
  
    This function uses an analytical formula to calculate the envelope of a family of parabolas 
  
    Parameters: 
    x : It forms the boundary of the domain for which the corresponding y-values of the envelope
        are to be calculated
    v : Initial velocity for which the envelope is to be drawn
    
    Returns: 
    x_vals : Array containning x values of the envelope
    y_vals : Array containning y values of the envelope

    Output:
    Plots the x_vals and y_vals to generate the envelope
    """
    
    x_vals = np.linspace(-x,x,n)
    y_vals = v**2/(2*g) - (g * x_vals**2)/(2*v**2)
    if show:
        py.plot(x_vals, y_vals, 'k--')
    return x_vals,y_vals




def f(x,v=1):
    """
    Outline. 
  
    This function uses the analytical formula for the envelope of parabola to calculate the 
    y value corresponding to given x and initial velocity 
  
    Parameters: 
    x : x value for which y value is to be calculated
    v : Initial velocity for which the y value is to be calculated
    
    
    Returns: 
    y : y value corresponding to x and v on the envelope       
    """
    y =  v**2/(2) - (x**2)/(2*v**2)
    return y
  
    

def volPara(N=1000,v=1):
    """
    Outline. 
  
    This function uses Monte-Carlo integration method to calculate the volume of safety for the 
    paraboloid formed by a ballistic object
    
  
    Parameters: 
    N : Number of random points to be generated 
    v : Initial velocity of the object to be launched
    
    Returns: 
    volume : The volume of conatined by the paraboloid    
    """
    
    x_max = np.sqrt((2*v**2)*(v**2/(2)))
    y_max = v**2/(2)
    x_vals, y_vals = envParabola(x_max, show = False)    
    x_rand = np.random.uniform(low=-x_max, high=x_max, size=(N,))
    y_rand = np.random.uniform(low=0, high=y_max, size=(N,))
    
    ind_below = np.where(y_rand < f(x_rand))
    ind_above = np.where(y_rand >= f(x_rand))
    
    py.scatter(x_rand[ind_below], y_rand[ind_below], color = "green")
    py.scatter(x_rand[ind_above], y_rand[ind_above], color = "blue")
    
    cross_area = (np.size(ind_below)/N)*2*x_max*y_max
    volume = np.pi*cross_area
    return volume



def targetCalcVerlet(theta,r, v_init=1):
    """
    Outline. 
  
    This function employs verlet routine to calculate the point on a spherical body(e.g planet)
    an object hits if launched from the north pole 
  
    Parameters: 
    theta  : Initial angle from which the projectile is launched
    r      : Radius of the spherical body(e.g planet)
    v_init : Initial velocity of the projectile   
    
    Returns: 
    r_xnew : x coordinate of the point projectile hits spherical body 
    r_ynew : y coordinate of the point projectile hits spherical body
    t      : time elapsed when projectile hits the spherical body
    
    If the velocity is greater than escape velocity and the body escapes the orbit
    it prints a message and returns none
    """
    if (theta<0 or theta>np.pi):
        print('Invalid input angle')
        return None,None,None
    
    dt = 0.0001
    t = 0
    r_xprev , r_yprev = 0.0 , r
    v_x , v_y = v_init * np.cos(theta), v_init * np.sin(theta)
    r_xnew , r_ynew = r_xprev + dt * v_x, r_yprev + dt * v_y  
    r_xdup , r_ydup = r_xnew , r_ynew 
    t = 2*dt
    while(r_xnew**2 + r_ynew**2 >= r**2):
        r_xnew = 2*r_xnew - r_xprev + dt**2 * ((-4*np.pi**2*r_xnew)/(r_xnew**2+r_ynew**2)**(3/2)) 
        r_ynew = 2*r_ynew - r_yprev + dt**2 * ((-4*np.pi**2*r_ynew)/(r_xnew**2+r_ynew**2)**(3/2))
        r_xprev, r_yprev = r_xdup , r_ydup 
        r_xdup , r_ydup = r_xnew , r_ynew 
        t += dt
        if ((r_xdup**2+r_ydup**2)>30*r**2):
            print('projectile escapes')
            return None,None,None
        
       
    return r_xnew, r_ynew ,t


       
    
def leapfrog(theta, r = 1.0,v_init=1, t_period = 1 ):
    """
    Outline. 
  
    This function employs leapfrog routine to calculate the elliptic trajectory 
    of an object under force from a point mass at the origin.
    The equations used are scaled, hence unitless.
  
    Parameters:  
    theta    : Initial angle by which the obeject is launced w.r.t positive x-axis
    v_init   : Initial velocity of the object
    t_period : Time period for which the simulation is run
    
    Returns: 
    r_x: Array containning x values of the object at dt time differences
    r_y: Array containning y values of the object at dt time differences    
    """
    
    dt = 0.0001
    period = t_period + dt
    length = int(period / dt)
    r_x = np.zeros(length)
    r_y = np.zeros(length)
    v_x = np.zeros(length)
    v_y = np.zeros(length)
    
    r_x[0] , r_y[0] = 0.0 , r
    v_x[0] , v_y[0] = v_init * np.cos(theta), v_init * np.sin(theta)
    r_x[1] , r_y[1] = r_x[0] + dt * v_x[0], r_y[0] + dt * v_y[0]  
    v_x[1] = v_x[0] + dt * ((-4*np.pi**2*r_x[0])/(r_x[0]**2+r_y[0]**2)**(3/2))
    v_y[1] = v_y[0] + dt * ((-4*np.pi**2*r_y[0])/(r_x[0]**2+r_y[0]**2)**(3/2)) 
    
    for i in range(2, length):
       
        r_x[i] = r_x[i-2] + 2 *dt* v_x[i-1] 
        r_y[i] = r_y[i-2] + 2 *dt* v_y[i-1]
        v_x[i] = v_x[i-2] + 2 *dt* ((-4*np.pi**2*r_x[i-1])/(r_x[i-1]**2+r_y[i-1]**2)**(3/2)) 
        v_y[i] = v_y[i-2] + 2 *dt* ((-4*np.pi**2*r_y[i-1])/(r_x[i-1]**2+r_y[i-1]**2)**(3/2))
        if (abs(r_x[i])< 10**-2 and abs(r_y[i])< 10**-2):
            return r_x, r_y 
 
    return r_x, r_y   



def envEllipse(v_init,r):
    """
    Outline. 
  
    This function uses the formula of ellipse with focal points at the origin where the point mass
    is located and the other at the point from where the object is launched to calculate the envelope
    of a family of ellipses 
  
    Parameters: 
    v_init : Initial velocity of the object
    r      : Vertical distance from point mass from where the object is launched
    
    Returns: 
    r_max : The vertical distance from the origin to the maximum y coordinate of the
            envelope

    Output:
    Plots the x and y to generate the envelope
    """
    r_max = r/(1-((v_init)/(np.sqrt(8/r)*np.pi))**2)
    theta = np.linspace(0,np.pi*2,100)    
    b = r_max - r/2
    a = np.sqrt((r_max-0.5*r)**2-(r/2)**2)
    x = a * np.cos(theta)
    y = 0.5*r + b * np.sin(theta)
    py.plot(x,y,'k--') 
    return r_max



def plotEllipse(v_ratio, t_period, r=2, theta_diff=np.pi/13):
    """
    Outline. 
  
    This function plots elliptical trajectories for an object with given function with a constant velocity
    but for varying angles
  
    Parameters: 
    v_ratio    : Ratio w.r.t the initial velocity required for a circular trajectory 
    t_period   : Time period for which the simulation is run 
    r          : Vertical distance from the origin where the object is launched
    theta_diff : The difference in launching angle between consecutive trajectories  
    
    Output: 
    It generates plots for trajectories with a constant velocity but different angles and
    the envelope for that family of trajectories 
    
    """
    arr = np.arange(0,2*np.pi,theta_diff)
    v_circ = 2*np.pi/np.sqrt(r)
    v_init = v_ratio*v_circ
    for i in arr:
        x,y = leapfrog(i,r,v_init, t_period)
        py.plot(x,y,'k-')
        py.plot(0,0,'r*')
        py.plot(0,r,'k*')
    envEllipse(v_init,r)



def fEllip(x,r=1,v_init=2*np.pi):
    """
    Outline. 
  
    This function uses the analytical formula for the envelope of ellipse to calculate the 
    y value corresponding to given x and initial velocity 
  
    Parameters: 
    x : x value for which y value is to be calculated
    r : Vertical distance from the origin where the object is launched
    v : Initial velocity for which the y value is to be calculated
    
    
    Returns: 
    y : y value corresponding to x, v and r on the envelope       
    """
    r_max = r/(1-((v_init)/(np.sqrt(8/r)*np.pi))**2)  
    b = r_max - r/2
    a = np.sqrt((r_max-0.5*r)**2-(r/2)**2)
    y = 0.5*r + b * np.sin(np.arccos(x/a))
    
    return y
    
def volEllipse(N=1000,r=1,v_init=2*np.pi):
    """
    Outline. 
  
    This function uses Monte-Carlo integration method to calculate the volume of safety for the 
    spheroid formed by a ballistic object
    
  
    Parameters: 
    N : Number of random points to be generated 
    r : Vertical distance from the origin where the object is launched
    v : Initial velocity of the object to be launched
    
    Returns: 
    volume : The volume of conatined by the spheroid   
    """
    r_max = r/(1-((v_init)/(np.sqrt(8/r)*np.pi))**2)  
    b = r_max - r/2
    a = np.sqrt((r_max-0.5*r)**2-(r/2)**2)   
    x_rand = np.random.uniform(low=-a, high=a, size=(N,))
    y_rand = np.random.uniform(low=r/2, high=b+r/2, size=(N,))
    py.plot(x_rand,y_rand,'*')
    
    ind_below = np.where(y_rand < abs(fEllip(x_rand)))
    ind_above = np.where(y_rand >= abs(fEllip(x_rand)))
    
    py.scatter(x_rand[ind_below], y_rand[ind_below], color = "green")
    py.scatter(x_rand[ind_above], y_rand[ind_above], color = "blue")
    
    cross_area = 2*(np.size(ind_below)/N)*2*a*b
    volume = np.pi*cross_area
    return volume    
    

