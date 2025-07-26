# The construction of the flux functions that we use here is presented in
# detail in A.J. Cerfon and J.P. Freidberg, ``One size fits all" analytic
# solutions to the Grad-Shafranov equation, Physics of Plasmas 17, 032502 (2010)

# MIT License

# Copyright (C) 2025: Antoine Cerfon
# Contact: antoine.cerfon@typeoneenergy.com
# 

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from ExactSolutions import *
import matplotlib.pyplot as plt

# Three equilibrium types available in this example:
#	- simple up-down symmetric equilibrium, associated with the string "symmetric"
#	- up-down symmetric equilibrium at the equilibrium beta limit, associated with the string "symmetric_beta_limit"
#	- up-down asymmetric equilibrium with a single-null point, associated with the string "asym_single_null"	
	
eq_type = "symmetric"

################################################################################
#
#   Specify equilibrium parameters of interest (in this example, we took
#   the ITER tokamak as an example)
#   
################################################################################

epsilon = 0.32
kappa = 1.7
delta = 0.33
A = -0.05 # Beta parameter - will not be used if A is computed self-consistently for an equilibrium at the equilibrium beta limit
xsep = .88;#x-location of the separatrix - will only be used for up-down asymmetric equilibria with a single null
ysep = -.6;#y-location of the separatrix - will only be used for up-down asymmetric equilibria with a single null


################################################################################

alpha = np.arcsin(delta);#alpha as defined in the article
slope1 = 0;#outer equatorial point slope
slope2 = 0;#inner equatorial point slope
curv1 = -(1+alpha)**2/(epsilon*kappa**2);#curvature at the outboard midplane
curv2 = -kappa/(epsilon*(np.cos(alpha))**2);#curvature at the top
curv3 = (1-alpha)**2/(epsilon*kappa**2);#curvature at the inboard midplane

match eq_type:
    case "symmetric":
            ################################################################################
        #
        #   Construct the matrix M of the boundary conditions for the funtions
        #   which are solutions to the homogeneous equation
        #   
        ################################################################################
        M = np.array([[psi1(1+epsilon,0),psi2(1+epsilon,0),psi3(1+epsilon,0),psi4(1+epsilon,0),psi5(1+epsilon,0),psi6(1+epsilon,0),psi7(1+epsilon,0)], #outer equatorial point
            [psi1(1-epsilon,0),psi2(1-epsilon,0),psi3(1-epsilon,0),psi4(1-epsilon,0),psi5(1-epsilon,0),psi6(1-epsilon,0),psi7(1-epsilon,0)], #inner equatorial point
            [psi1(1-epsilon*delta,kappa*epsilon),psi2(1-epsilon*delta,kappa*epsilon),psi3(1-epsilon*delta,kappa*epsilon),psi4(1-epsilon*delta,kappa*epsilon),psi5(1-epsilon*delta,kappa*epsilon),psi6(1-epsilon*delta,kappa*epsilon),psi7(1-epsilon*delta,kappa*epsilon)], #upper high point
            [psi1x(1-epsilon*delta,kappa*epsilon),psi2x(1-epsilon*delta,kappa*epsilon),psi3x(1-epsilon*delta,kappa*epsilon), psi4x(1-epsilon*delta,kappa*epsilon),psi5x(1-epsilon*delta,kappa*epsilon),psi6x(1-epsilon*delta,kappa*epsilon),psi7x(1-epsilon*delta,kappa*epsilon)], #upper high point maximum
            [curv1*psi1x(1+epsilon,0)+psi1yy(1+epsilon,0),curv1*psi2x(1+epsilon,0)+psi2yy(1+epsilon,0),curv1*psi3x(1+epsilon,0)+psi3yy(1+epsilon,0),curv1*psi4x(1+epsilon,0)+psi4yy(1+epsilon,0),curv1*psi5x(1+epsilon,0)+psi5yy(1+epsilon,0), curv1*psi6x(1+epsilon,0)+psi6yy(1+epsilon,0),curv1*psi7x(1+epsilon,0)+psi7yy(1+epsilon,0)], #curvature condition at outer equatorial point
            [curv3*psi1x(1-epsilon,0)+psi1yy(1-epsilon,0),curv3*psi2x(1-epsilon,0)+psi2yy(1-epsilon,0),curv3*psi3x(1-epsilon,0)+psi3yy(1-epsilon,0),curv3*psi4x(1-epsilon,0)+psi4yy(1-epsilon,0),curv3*psi5x(1-epsilon,0)+psi5yy(1-epsilon,0), curv3*psi6x(1-epsilon,0)+psi6yy(1-epsilon,0),curv3*psi7x(1-epsilon,0)+psi7yy(1-epsilon,0)], #curvature condition at inner equatorial point
            [curv2*psi1y(1-epsilon*delta,kappa*epsilon)+psi1xx(1-epsilon*delta,kappa*epsilon),curv2*psi2y(1-epsilon*delta,kappa*epsilon)+psi2xx(1-epsilon*delta,kappa*epsilon),curv2*psi3y(1-epsilon*delta,kappa*epsilon)+psi3xx(1-epsilon*delta,kappa*epsilon),curv2*psi4y(1-epsilon*delta,kappa*epsilon)+psi4xx(1-epsilon*delta,kappa*epsilon), curv2*psi5y(1-epsilon*delta,kappa*epsilon)+psi5xx(1-epsilon*delta,kappa*epsilon),curv2*psi6y(1-epsilon*delta,kappa*epsilon)+psi6xx(1-epsilon*delta,kappa*epsilon),curv2*psi7y(1-epsilon*delta,kappa*epsilon)+psi7xx(1-epsilon*delta,kappa*epsilon)]]) #curvature condition at top

################################################################################
#
#   Construct the vector b of the boundary conditions for the particular
#   solutions to the equation
#
#
################################################################################

        b = -np.array([[A*psipart1(1+epsilon,0)+(1-A)*psipart2(1+epsilon,0)], # outer equatorial point
            [A*psipart1(1-epsilon,0)+(1-A)*psipart2(1-epsilon,0)], # inner equatorial point
            [A*psipart1(1-epsilon*delta,kappa*epsilon)+(1-A)*psipart2(1-epsilon*delta,kappa*epsilon)], #upper high point
            [A*psipart1x(1-epsilon*delta,kappa*epsilon)+(1-A)*psipart2x(1-epsilon*delta,kappa*epsilon)], #upper high point maximum
            [A*(curv1*psipart1x(1+epsilon,0)+psipart1yy(1+epsilon,0))+(1-A)*(curv1*psipart2x(1+epsilon,0)+psipart2yy(1+epsilon,0))],#curvature condition at outer equatorial point
            [A*(curv3*psipart1x(1-epsilon,0)+psipart1yy(1-epsilon,0))+(1-A)*(curv3*psipart2x(1-epsilon,0)+psipart2yy(1-epsilon,0))],#curvature condition at inner equatorial point
            [A*(curv2*psipart1y(1-epsilon*delta,kappa*epsilon)+psipart1xx(1-epsilon*delta,kappa*epsilon))+(1-A)*(curv2*psipart2y(1-epsilon*delta,kappa*epsilon)+psipart2xx(1-epsilon*delta,kappa*epsilon))]])#curvature condition at top

        ################################################################################
        #
        #   Solve the linear system for the coefficients C of the general
        #   solution to the equation
        #
        ################################################################################

        C = np.linalg.solve(M, b)
        # Pad the coefficients with zeros for the up-down asymmetric terms
        C=np.concatenate((C,np.zeros((5,1))))
        ysep = -kappa*epsilon
        
        contour_levels = np.linspace(-0.045,0,20)
        
    case "symmetric_beta_limit":
            ################################################################################
        #
        #   Construct the matrix M of the boundary conditions for the funtions
        #   which are solutions to the homogeneous equation
        #   
        ################################################################################
        M = np.array([[psi1(1+epsilon,0),psi2(1+epsilon,0),psi3(1+epsilon,0),psi4(1+epsilon,0),psi5(1+epsilon,0),psi6(1+epsilon,0),psi7(1+epsilon,0),psipart1(1+epsilon,0)-psipart2(1+epsilon,0)], #outer equatorial point
            [psi1(1-epsilon,0),psi2(1-epsilon,0),psi3(1-epsilon,0),psi4(1-epsilon,0),psi5(1-epsilon,0),psi6(1-epsilon,0),psi7(1-epsilon,0),psipart1(1-epsilon,0)-psipart2(1-epsilon,0)], #inner equatorial point
            [psi1(1-epsilon*delta,kappa*epsilon),psi2(1-epsilon*delta,kappa*epsilon),psi3(1-epsilon*delta,kappa*epsilon),psi4(1-epsilon*delta,kappa*epsilon),psi5(1-epsilon*delta,kappa*epsilon),psi6(1-epsilon*delta,kappa*epsilon),psi7(1-epsilon*delta,kappa*epsilon),psipart1(1-epsilon*delta,kappa*epsilon)-psipart2(1-epsilon*delta,kappa*epsilon)], #upper high point
            [psi1x(1-epsilon*delta,kappa*epsilon),psi2x(1-epsilon*delta,kappa*epsilon),psi3x(1-epsilon*delta,kappa*epsilon), psi4x(1-epsilon*delta,kappa*epsilon),psi5x(1-epsilon*delta,kappa*epsilon),psi6x(1-epsilon*delta,kappa*epsilon),psi7x(1-epsilon*delta,kappa*epsilon),psipart1x(1-epsilon*delta,kappa*epsilon)-psipart2x(1-epsilon*delta,kappa*epsilon)], #upper high point maximum
            [curv1*psi1x(1+epsilon,0)+psi1yy(1+epsilon,0),curv1*psi2x(1+epsilon,0)+psi2yy(1+epsilon,0),curv1*psi3x(1+epsilon,0)+psi3yy(1+epsilon,0),curv1*psi4x(1+epsilon,0)+psi4yy(1+epsilon,0),curv1*psi5x(1+epsilon,0)+psi5yy(1+epsilon,0), curv1*psi6x(1+epsilon,0)+psi6yy(1+epsilon,0),curv1*psi7x(1+epsilon,0)+psi7yy(1+epsilon,0),curv1*(psipart1x(1+epsilon,0)-psipart2x(1+epsilon,0))+psipart1yy(1+epsilon,0)-psipart2yy(1+epsilon,0)], #curvature condition at outer equatorial point
            [curv3*psi1x(1-epsilon,0)+psi1yy(1-epsilon,0),curv3*psi2x(1-epsilon,0)+psi2yy(1-epsilon,0),curv3*psi3x(1-epsilon,0)+psi3yy(1-epsilon,0),curv3*psi4x(1-epsilon,0)+psi4yy(1-epsilon,0),curv3*psi5x(1-epsilon,0)+psi5yy(1-epsilon,0), curv3*psi6x(1-epsilon,0)+psi6yy(1-epsilon,0),curv3*psi7x(1-epsilon,0)+psi7yy(1-epsilon,0),curv3*(psipart1x(1-epsilon,0)-psipart2x(1-epsilon,0))+psipart1yy(1-epsilon,0)-psipart2yy(1-epsilon,0)], #curvature condition at inner equatorial point
            [curv2*psi1y(1-epsilon*delta,kappa*epsilon)+psi1xx(1-epsilon*delta,kappa*epsilon),curv2*psi2y(1-epsilon*delta,kappa*epsilon)+psi2xx(1-epsilon*delta,kappa*epsilon),curv2*psi3y(1-epsilon*delta,kappa*epsilon)+psi3xx(1-epsilon*delta,kappa*epsilon),curv2*psi4y(1-epsilon*delta,kappa*epsilon)+psi4xx(1-epsilon*delta,kappa*epsilon), curv2*psi5y(1-epsilon*delta,kappa*epsilon)+psi5xx(1-epsilon*delta,kappa*epsilon),curv2*psi6y(1-epsilon*delta,kappa*epsilon)+psi6xx(1-epsilon*delta,kappa*epsilon),curv2*psi7y(1-epsilon*delta,kappa*epsilon)+psi7xx(1-epsilon*delta,kappa*epsilon),curv2*(psipart1y(1-epsilon*delta,kappa*epsilon)-psipart2y(1-epsilon*delta,kappa*epsilon))+psipart1xx(1-epsilon*delta,kappa*epsilon)-psipart2xx(1-epsilon*delta,kappa*epsilon)],#curvature condition at top
            [psi1x(1-epsilon,0),psi2x(1-epsilon,0),psi3x(1-epsilon,0),psi4x(1-epsilon,0),psi5x(1-epsilon,0),psi6x(1-epsilon,0),psi7x(1-epsilon,0),psipart1x(1-epsilon,0)-psipart2x(1-epsilon,0)]]) #Equilibrium beta limit condition

################################################################################
#
#   Construct the vector b of the boundary conditions for the particular
#   solutions to the equation
#
#
################################################################################

        b = -np.array([[psipart2(1+epsilon,0)], # outer equatorial point
            [psipart2(1-epsilon,0)], # inner equatorial point
            [psipart2(1-epsilon*delta,kappa*epsilon)], #upper high point
            [psipart2x(1-epsilon*delta,kappa*epsilon)], #upper high point maximum
            [curv1*psipart2x(1+epsilon,0)+psipart2yy(1+epsilon,0)],#curvature condition at outer equatorial point
            [curv3*psipart2x(1-epsilon,0)+psipart2yy(1-epsilon,0)],#curvature condition at inner equatorial point
            [curv2*psipart2y(1-epsilon*delta,kappa*epsilon)+psipart2xx(1-epsilon*delta,kappa*epsilon)],#curvature condition at top
            [psipart2x(1-epsilon,0)]])#Equilibrium beta limit condition

        ################################################################################
        #
        #   Solve the linear system for the coefficients C of the general
        #   solution to the equation
        #
        ################################################################################

        C = np.linalg.solve(M, b)
        A = C[7]
        
        # Pad the coefficients with zeros for the up-down asymmetric terms
        C=np.concatenate((C[0:7],np.zeros((5,1))))
        ysep = -kappa*epsilon
        
        contour_levels = np.linspace(-0.045,0,20)
        
    case "asym_single_null":
        ################################################################################
        #
        #   Construct the matrix M of the boundary conditions for the funtions
        #   which are solutions to the homogeneous equation
        #   
        ################################################################################
        M = np.array([[psi1(1+epsilon,0),psi2(1+epsilon,0),psi3(1+epsilon,0),psi4(1+epsilon,0),psi5(1+epsilon,0),psi6(1+epsilon,0),psi7(1+epsilon,0),psi8(1+epsilon,0),psi9(1+epsilon,0),psi10(1+epsilon,0),psi11(1+epsilon,0),psi12(1+epsilon,0)], #outer equatorial point
            [psi1(1-epsilon,0),psi2(1-epsilon,0),psi3(1-epsilon,0),psi4(1-epsilon,0),psi5(1-epsilon,0),psi6(1-epsilon,0),psi7(1-epsilon,0),psi8(1-epsilon,0),psi9(1-epsilon,0),psi10(1-epsilon,0),psi11(1-epsilon,0),psi12(1-epsilon,0)], #inner equatorial point
            [psi1(1-epsilon*delta,kappa*epsilon),psi2(1-epsilon*delta,kappa*epsilon),psi3(1-epsilon*delta,kappa*epsilon),psi4(1-epsilon*delta,kappa*epsilon),psi5(1-epsilon*delta,kappa*epsilon),psi6(1-epsilon*delta,kappa*epsilon),psi7(1-epsilon*delta,kappa*epsilon),psi8(1-epsilon*delta,kappa*epsilon),psi9(1-epsilon*delta,kappa*epsilon),psi10(1-epsilon*delta,kappa*epsilon),psi11(1-epsilon*delta,kappa*epsilon),psi12(1-epsilon*delta,kappa*epsilon)], #upper high point
            [psi1(xsep,ysep),psi2(xsep,ysep),psi3(xsep,ysep),psi4(xsep,ysep),psi5(xsep,ysep),psi6(xsep,ysep),psi7(xsep,ysep), psi8(xsep,ysep),psi9(xsep,ysep),psi10(xsep,ysep),psi11(xsep,ysep),psi12(xsep,ysep)], #lower X point    
            [slope1*psi1x(1+epsilon,0)+psi1y(1+epsilon,0),slope1*psi2x(1+epsilon,0)+psi2y(1+epsilon,0),slope1*psi3x(1+epsilon,0)+psi3y(1+epsilon,0),slope1*psi4x(1+epsilon,0)+psi4y(1+epsilon,0),slope1*psi5x(1+epsilon,0)+psi5y(1+epsilon,0), slope1*psi6x(1+epsilon,0)+psi6y(1+epsilon,0),slope1*psi7x(1+epsilon,0)+psi7y(1+epsilon,0),slope1*psi8x(1+epsilon,0)+psi8y(1+epsilon,0),slope1*psi9x(1+epsilon,0)+psi9y(1+epsilon,0),slope1*psi10x(1+epsilon,0)+psi10y(1+epsilon,0),slope1*psi11x(1+epsilon,0)+psi11y(1+epsilon,0),slope1*psi12x(1+epsilon,0)+psi12y(1+epsilon,0)], #outer equatorial point slope
            [slope2*psi1x(1-epsilon,0)+psi1y(1-epsilon,0),slope2*psi2x(1-epsilon,0)+psi2y(1-epsilon,0),slope2*psi3x(1-epsilon,0)+psi3y(1-epsilon,0),slope2*psi4x(1-epsilon,0)+psi4y(1-epsilon,0),slope2*psi5x(1-epsilon,0)+psi5y(1-epsilon,0), slope2*psi6x(1-epsilon,0)+psi6y(1-epsilon,0),slope2*psi7x(1-epsilon,0)+psi7y(1-epsilon,0),slope2*psi8x(1-epsilon,0)+psi8y(1-epsilon,0),slope2*psi9x(1-epsilon,0)+psi9y(1-epsilon,0),slope2*psi10x(1-epsilon,0)+psi10y(1-epsilon,0),slope2*psi11x(1-epsilon,0)+psi11y(1-epsilon,0),slope2*psi12x(1-epsilon,0)+psi12y(1-epsilon,0)], #inner equatorial point slope
            [psi1x(1-epsilon*delta,kappa*epsilon),psi2x(1-epsilon*delta,kappa*epsilon),psi3x(1-epsilon*delta,kappa*epsilon), psi4x(1-epsilon*delta,kappa*epsilon),psi5x(1-epsilon*delta,kappa*epsilon),psi6x(1-epsilon*delta,kappa*epsilon),psi7x(1-epsilon*delta,kappa*epsilon),psi8x(1-epsilon*delta,kappa*epsilon),psi9x(1-epsilon*delta,kappa*epsilon),psi10x(1-epsilon*delta,kappa*epsilon),psi11x(1-epsilon*delta,kappa*epsilon),psi12x(1-epsilon*delta,kappa*epsilon)], #upper high point maximum
            [psi1x(xsep,ysep),psi2x(xsep,ysep),psi3x(xsep,ysep),psi4x(xsep,ysep),psi5x(xsep,ysep),psi6x(xsep,ysep), psi7x(xsep,ysep),psi8x(xsep,ysep),psi9x(xsep,ysep),psi10x(xsep,ysep),psi11x(xsep,ysep),psi12x(xsep,ysep)], #By = 0 at lower X-point
            [psi1y(xsep,ysep),psi2y(xsep,ysep),psi3y(xsep,ysep),psi4y(xsep,ysep),psi5y(xsep,ysep),psi6y(xsep,ysep), psi7y(xsep,ysep),psi8y(xsep,ysep),psi9y(xsep,ysep),psi10y(xsep,ysep),psi11y(xsep,ysep),psi12y(xsep,ysep)], #Bx = 0 at lower X-point
            [curv1*psi1x(1+epsilon,0)+psi1yy(1+epsilon,0),curv1*psi2x(1+epsilon,0)+psi2yy(1+epsilon,0),curv1*psi3x(1+epsilon,0)+psi3yy(1+epsilon,0),curv1*psi4x(1+epsilon,0)+psi4yy(1+epsilon,0),curv1*psi5x(1+epsilon,0)+psi5yy(1+epsilon,0), curv1*psi6x(1+epsilon,0)+psi6yy(1+epsilon,0),curv1*psi7x(1+epsilon,0)+psi7yy(1+epsilon,0),curv1*psi8x(1+epsilon,0)+psi8yy(1+epsilon,0),curv1*psi9x(1+epsilon,0)+psi9yy(1+epsilon,0),curv1*psi10x(1+epsilon,0)+psi10yy(1+epsilon,0), curv1*psi11x(1+epsilon,0)+psi11yy(1+epsilon,0),curv1*psi12x(1+epsilon,0)+psi12yy(1+epsilon,0)], #curvature condition at outer equatorial point
            [curv3*psi1x(1-epsilon,0)+psi1yy(1-epsilon,0),curv3*psi2x(1-epsilon,0)+psi2yy(1-epsilon,0),curv3*psi3x(1-epsilon,0)+psi3yy(1-epsilon,0),curv3*psi4x(1-epsilon,0)+psi4yy(1-epsilon,0),curv3*psi5x(1-epsilon,0)+psi5yy(1-epsilon,0), curv3*psi6x(1-epsilon,0)+psi6yy(1-epsilon,0),curv3*psi7x(1-epsilon,0)+psi7yy(1-epsilon,0),curv3*psi8x(1-epsilon,0)+psi8yy(1-epsilon,0),curv3*psi9x(1-epsilon,0)+psi9yy(1-epsilon,0),curv3*psi10x(1-epsilon,0)+psi10yy(1-epsilon,0), curv3*psi11x(1-epsilon,0)+psi11yy(1-epsilon,0),curv3*psi12x(1-epsilon,0)+psi12yy(1-epsilon,0)], #curvature condition at inner equatorial point
            [curv2*psi1y(1-epsilon*delta,kappa*epsilon)+psi1xx(1-epsilon*delta,kappa*epsilon),curv2*psi2y(1-epsilon*delta,kappa*epsilon)+psi2xx(1-epsilon*delta,kappa*epsilon),curv2*psi3y(1-epsilon*delta,kappa*epsilon)+psi3xx(1-epsilon*delta,kappa*epsilon),curv2*psi4y(1-epsilon*delta,kappa*epsilon)+psi4xx(1-epsilon*delta,kappa*epsilon), curv2*psi5y(1-epsilon*delta,kappa*epsilon)+psi5xx(1-epsilon*delta,kappa*epsilon),curv2*psi6y(1-epsilon*delta,kappa*epsilon)+psi6xx(1-epsilon*delta,kappa*epsilon),curv2*psi7y(1-epsilon*delta,kappa*epsilon)+psi7xx(1-epsilon*delta,kappa*epsilon),curv2*psi8y(1-epsilon*delta,kappa*epsilon)+psi8xx(1-epsilon*delta,kappa*epsilon), curv2*psi9y(1-epsilon*delta,kappa*epsilon)+psi9xx(1-epsilon*delta,kappa*epsilon),curv2*psi10y(1-epsilon*delta,kappa*epsilon)+psi10xx(1-epsilon*delta,kappa*epsilon),curv2*psi11y(1-epsilon*delta,kappa*epsilon)+psi11xx(1-epsilon*delta,kappa*epsilon),curv2*psi12y(1-epsilon*delta,kappa*epsilon)+psi12xx(1-epsilon*delta,kappa*epsilon)]]) #curvature condition at top

################################################################################
#
#   Construct the vector b of the boundary conditions for the particular
#   solutions to the equation
#
#
################################################################################

        b = -np.array([[A*psipart1(1+epsilon,0)+(1-A)*psipart2(1+epsilon,0)], # outer equatorial point
            [A*psipart1(1-epsilon,0)+(1-A)*psipart2(1-epsilon,0)], # inner equatorial point
            [A*psipart1(1-epsilon*delta,kappa*epsilon)+(1-A)*psipart2(1-epsilon*delta,kappa*epsilon)], #upper high point
            [A*psipart1(xsep,ysep)+(1-A)*psipart2(xsep,ysep)], #lower X-point
            [A*(slope1*psipart1x(1+epsilon,0)+psipart1y(1+epsilon,0))+(1-A)*(slope1*psipart2x(1+epsilon,0)+psipart2y(1+epsilon,0))], #outer equatorial point slope
            [A*(slope2*psipart1x(1-epsilon,0)+psipart1y(1-epsilon,0))+(1-A)*(slope2*psipart2x(1-epsilon,0)+psipart2y(1-epsilon,0))], #inner equatorial point slope
            [A*psipart1x(1-epsilon*delta,kappa*epsilon)+(1-A)*psipart2x(1-epsilon*delta,kappa*epsilon)], #upper high point maximum
            [A*psipart1x(xsep,ysep)+(1-A)*psipart2x(xsep,ysep)], #By = 0 at lower X-point
            [A*psipart1y(xsep,ysep)+(1-A)*psipart2y(xsep,ysep)], #Bx = 0 at lower X-point
            [A*(curv1*psipart1x(1+epsilon,0)+psipart1yy(1+epsilon,0))+(1-A)*(curv1*psipart2x(1+epsilon,0)+psipart2yy(1+epsilon,0))],#curvature condition at outer equatorial point
            [A*(curv3*psipart1x(1-epsilon,0)+psipart1yy(1-epsilon,0))+(1-A)*(curv3*psipart2x(1-epsilon,0)+psipart2yy(1-epsilon,0))],#curvature condition at inner equatorial point
            [A*(curv2*psipart1y(1-epsilon*delta,kappa*epsilon)+psipart1xx(1-epsilon*delta,kappa*epsilon))+(1-A)*(curv2*psipart2y(1-epsilon*delta,kappa*epsilon)+psipart2xx(1-epsilon*delta,kappa*epsilon))]])#curvature condition at top

        ################################################################################
        #
        #   Solve the linear system for the coefficients C of the general
        #   solution to the equation
        #
        ################################################################################

        C = np.linalg.solve(M, b)
        
        contour_levels = np.linspace(-0.045,0,20)

################################################################################
#
#   Compute the poloidal flux function
#
################################################################################

x = np.linspace(1-epsilon-0.05, 1+epsilon+0.1, 2000)
y = np.linspace(ysep-0.05, kappa*epsilon+0.025, 2000)

X, Y = np.meshgrid(x, y)

Z = C[0]*psi1(X,Y)+C[1]*psi2(X,Y)+C[2]*psi3(X,Y)+C[3]*psi4(X,Y)+C[4]*psi5(X,Y) \
   +C[5]*psi6(X,Y)+C[6]*psi7(X,Y)+C[7]*psi8(X,Y)+C[8]*psi9(X,Y) \
   +C[9]*psi10(X,Y)+C[10]*psi11(X,Y)+C[11]*psi12(X,Y) \
   +A*psipart1(X,Y)+(1-A)*psipart2(X,Y)
        
cmap = plt.get_cmap('copper_r')
   
h = plt.contour(X, Y, Z, levels=contour_levels)
plt.axvline(x=0.0, linestyle = '--',color='black')
plt.xlabel("$R/R_{0}$",fontsize = 20)
plt.ylabel("$Z/R_{0}$",fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.axis('equal')
plt.xlim(0,1+epsilon+0.25)
plt.ylim(ysep-0.2, kappa*epsilon+0.2)
plt.set_cmap(cmap)
#plt.colorbar()
plt.show()
