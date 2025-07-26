# In this file, we define all the special solutions and the homogeneous solutions to the Grad-Shafranov equation as defined in A.J. Cerfon and J.P. Freidberg, “One size fits all” analytic solutions to the Grad–Shafranov equation, Physics of Plasmas 17, 032502 (2010)

import numpy as np

# psi 1 and all its derivatives

def psi1(x,y):
   return 1
   
def psi1x(x,y):
   return 0

def psi1xx(x,y):
   return 0
   
def psi1y(x,y):
   return 0

def psi1yy(x,y):
   return 0

# psi 2 and all its derivatives

def psi2(x,y):
   return x**2
   
def psi2x(x,y):
   return 2*x

def psi2xx(x,y):
   return 2
   
def psi2y(x,y):
   return 0

def psi2yy(x,y):
   return 0

# psi 3 and all its derivatives

def psi3(x,y):
   return y**2-x**2*np.log(x)
   
def psi3x(x,y):
   return -2*x*np.log(x)-x

def psi3xx(x,y):
   return -2*np.log(x)-3
   
def psi3y(x,y):
   return 2*y

def psi3yy(x,y):
   return 2
   
# psi 4 and all its derivatives

def psi4(x,y):
   return x**4-4*x**2*y**2
   
def psi4x(x,y):
   return 4*x**3-8*x*y**2

def psi4xx(x,y):
   return 12*x**2-8*y**2
   
def psi4y(x,y):
   return -8*x**2*y

def psi4yy(x,y):
   return -8*x**2
   
# psi 5 and all its derivatives

def psi5(x,y):
   return 2*y**4-9*y**2*x**2+3*x**4*np.log(x)-12*x**2*y**2*np.log(x)
   
def psi5x(x,y):
   return -30*x*y**2+12*x**3*np.log(x)+3*x**3-24*x*np.log(x)*y**2

def psi5xx(x,y):
   return -54*y**2+36*x**2*np.log(x)+21*x**2-24*np.log(x)*y**2
   
def psi5y(x,y):
   return 8*y**3-18*y*x**2-24*x**2*y*np.log(x)

def psi5yy(x,y):
   return 24*y**2-18*x**2-24*x**2*np.log(x)
   x
# psi 6 and all its derivatives

def psi6(x,y):
   return x**6-12*x**4*y**2+8*x**2*y**4
   
def psi6x(x,y):
   return 6*x**5-48*x**3*y**2+16*x*y**4

def psi6xx(x,y):
   return 30*x**4-144*x**2*y**2+16*y**4
   
def psi6y(x,y):
   return -24*x**4*y+32*x**2*y**3

def psi6yy(x,y):
   return -24*x**4+96*x**2*y**2
   
# psi 7 and all its derivatives

def psi7(x,y):
   return 8*y**6-140*y**4*x**2+75*y**2*x**4-15*x**6*np.log(x)+180*x**4*y**2*np.log(x)-120*x**2*y**4*np.log(x)
   
def psi7x(x,y):
   return -400*x*y**4+480*x**3*y**2-90*x**5*np.log(x)+720*y**2*x**3*np.log(x)-15*x**5-240*y**4*x*np.log(x)

def psi7xx(x,y):
   return -640*y**4+2160*x**2*y**2-450*x**4*np.log(x)-165*x**4+2160*y**2*x**2*np.log(x)-240*y**4*np.log(x)
   
def psi7y(x,y):
   return 48*y**5-560*x**2*y**3-480*x**2*np.log(x)*y**3+360*x**4*np.log(x)*y+150*x**4*y

def psi7yy(x,y):
   return 240*y**4-1680*x**2*y**2-1440*x**2*np.log(x)*y**2+360*x**4*np.log(x)+150*x**4
   
def psi8(x,y):
   return y

def psi8x(x,y):
   return 0

def psi8xx(x,y):
   return 0

def psi8y(x,y):
   return 1

def psi8yy(x,y):
   return 0

def psi9(x,y):
   return y*x**2

def psi9x(x,y):
   return 2*y*x

def psi9xx(x,y):
   return 2*y

def psi9y(x,y):
   return x**2

def psi9yy(x,y):
   return 0
   
def psi10(x,y):
   return y**3-3*y*x**2*np.log(x)

def psi10x(x,y):
   return -6*y*x*np.log(x)-3*y*x

def psi10xx(x,y):
   return -6*y*np.log(x)-9*y

def psi10y(x,y):
   return 3*y**2-3*x**2*np.log(x)

def psi10yy(x,y):
   return 6*y
   
def psi11(x,y):
   return 3*y*x**4-4*y**3*x**2

def psi11x(x,y):
   return 12*y*x**3-8*y**3*x

def psi11xx(x,y):
   return 36*y*x**2-8*y**3

def psi11y(x,y):
   return 3*x**4-12*y**2*x**2

def psi11yy(x,y):
   return -24*y*x**2
   
def psi12(x,y):
   return 8*y**5-45*y*x**4-80*y**3*x**2*np.log(x)+60*y*x**4*np.log(x)

def psi12x(x,y):
   return -120*y*x**3-160*y**3*x*np.log(x)+240*y*x**3*np.log(x)-80*y**3*x

def psi12xx(x,y):
   return -120*y*x**2-160*y**3*np.log(x)+720*y*x**2*np.log(x)-240*y**3

def psi12y(x,y):
   return 40*y**4-45*x**4-240*y**2*x**2*np.log(x)+60*x**4*np.log(x)

def psi12yy(x,y):
   return 160*y**3-480*y*x**2*np.log(x)   

def psipart1(x,y):
   return 1/2*x**2*np.log(x)
   
def psipart1x(x,y):
   return x*np.log(x)+x/2

def psipart1xx(x,y):
   return np.log(x)+3/2
   
def psipart1y(x,y):
   return 0

def psipart1yy(x,y):
   return 0

def psipart2(x,y):
   return x**4/8
   
def psipart2x(x,y):
   return x**3/2

def psipart2xx(x,y):
   return 3*x**2/2
   
def psipart2y(x,y):
   return 0

def psipart2yy(x,y):
   return 0 


