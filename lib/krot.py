

# Kantorovich-Rubenstein Optimal Transport

import cvxpy as cp
from lib.findiff import *
import matplotlib.pyplot as plt 

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
DPI = 160

def lin_krot(x, rho, sigma, Gc, diff = 'ForwardEuler', vis=False):
    FD = {'ForwardEuler' : D1FE, 'BackwardEuler' : D1BE, 'Centered': D1CD}
    n = len(x)
    h = x[1]-x[0]

    c = rho-sigma
    A = FD[diff](n, bc = 'Neumann',matrix=True)/h
    b = Gc

    phi = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(c.T@phi), [A@phi<=b,-A@phi<=b])
    prob.solve()

    if vis:
        plt.figure(figsize=(10,10), dpi=DPI)
        plt.tight_layout()
        plt.subplot(221)
        plt.bar(x,rho, width = 3*h, label='$\\rho(x)$', color='blue')
        plt.bar(x,sigma, width = 3*h, label='$\sigma(x)$', color='red')
        plt.legend()
        plt.subplot(222)
        plt.plot(x,Gc, color='blue')
        plt.ylim(0,2.5)
        plt.title("$\\nabla c$")
        plt.subplot(223)
        plt.plot(x,phi.value, color='blue')
        plt.title("$\phi$")
        plt.subplot(224)
        plt.plot(x,A@phi.value,color='blue')
        plt.title("$\\nabla \phi$")
        plt.savefig("../img/lindist.pdf")
        plt.show()

    return phi.value, A@phi.value


def FDM_krot(x, rho, sigma, Gc, diff = 'ForwardEuler', vis=False):

    FD = {'ForwardEuler' : D1FE, 'BackwardEuler' : D1BE, 'Centered': D1CD}

    n = len(x)
    h=x[1]-x[0]

    #Linear System
    f = sigma-rho
    D = FD[diff](n, bc = 'Neumann',matrix=True)

    #First Step
    lam_t = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(D.T@lam_t-f)))
    prob.solve()
    #Second Step
    lam = abs(lam_t.value)
    Gphi = np.array([np.sign(np.round(lam_t.value[i],2))*Gc[i] for i in range(n)])
    phi = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(D@phi-Gphi)))
    prob.solve()


    if vis:
        plt.figure(figsize=(10,10))
        plt.tight_layout()
        plt.subplot(221)
        plt.bar(x,rho, width = 3*h, label='$\\rho(x)$',color='blue')
        plt.bar(x,sigma, width = 3*h, label='$\sigma(x)$',color='red')
        plt.legend()
        plt.subplot(222)
        plt.plot(x,Gc,color='blue')
        plt.ylim(0,2.5)
        plt.title("$\\nabla c$")
        plt.subplot(223)
        plt.plot(x,phi.value,color='blue')
        plt.title("$\phi$")
        plt.subplot(224)
        plt.plot(x,D@phi.value, label="$\\nabla \phi$", color='blue')
        plt.plot(x,lam,label="$\\lambda$", color='red')
        plt.ylim(-2.5,2.5)
        plt.title("$\\lambda, \\nabla \phi$")
        plt.legend()
        plt.savefig("../img/graddist.pdf")
        plt.show()

    return phi.value, D@phi.value, lam


def lambdaApprox(xx, yy, rho, sigma, GC, lam = None, diff='ForwardEuler', vis=False, alpha = 1.0, beta=1.0):
    #GRID
    n,m = rho.shape
    hx=xx[0,1]-xx[0,0]
    hy=yy[1,0]-yy[0,0]
    f = sigma.flatten()-rho.flatten()
    if lam is None:
        lam = np.ones((n,m))
    lam_diag = np.diag(np.append(lam.flatten(), lam.flatten()))
    BC = "Neumann"

    #DIFFERENCE SCHEMES
    FD = {'ForwardEuler' : [D2xFE,D2yFE], 'BackwardEuler' : [D2xBE,D2yBE], 'Centered': [D2xCD,D2yCD]}
    Dx = FD[diff][0](n,m, bc = BC, matrix=True)/hx
    Dy = FD[diff][1](n,m, bc = BC, matrix=True)/hy
    Div = np.hstack((Dx.T,Dy.T))
    Grad = np.vstack((Dx,Dy))

    #FIRST STEP
    lam_t = cp.Variable(n*m*2)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Div@lam_diag@lam_t-f)))
    prob.solve()
    lam_t = lam_t.value.reshape(2,n*n)

    #SCALE LAMBDA
    norm = np.array([np.linalg.norm(_lam_t) for _lam_t in lam_t.T])
    lam = norm
    lam_p=lam
    lam = np.diag(np.append(lam,lam))

    #SECOND STEP
    A = (lam@Grad)
    phi = cp.Variable(n*n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A@phi-lam_t.reshape(2*n*n))))
    prob.solve()
    phi = phi.value.reshape(n,n)
    Gphi = (Grad@phi.reshape(n*n)).reshape(2,n*n)

    norm = np.array([np.linalg.norm(gphi) for gphi in Gphi.T])

    lam = HarmAvg(norm.reshape((n,n)))

    lam_diag = np.diag(np.append(lam,lam))

    #PLOTTING
    if vis:
        plt.figure(figsize=(16,10))
        plt.subplot(221)
        plt.quiver(xx,yy,lam_t[0],lam_t[1], color='r')
        plt.title('First Iteration $\\lambda_t$')
        plt.subplot(222)
        plt.imshow(lam_p.reshape(n,n)[::-1])
        plt.title('First Iteration $\\lambda$')
        plt.colorbar()
        plt.subplot(223)
        plt.title('Second Iteration $\\lambda$')
        plt.imshow(lam[::-1])
        plt.subplot(224)
        plt.title('$\\nabla\\phi$')
        plt.quiver(xx,yy,Gphi[0], Gphi[1],color='r')
        plt.show()

    return lam,phi,Gphi



def lambdaHA(xx, yy, rho, sigma, GC, lam = None, diff='ForwardEuler', vis=False, alpha = 1.0, beta=1.0):
    #GRID
    n,m = rho.shape
    hx=xx[0,1]-xx[0,0]
    hy=yy[1,0]-yy[0,0]
    f = sigma.flatten()-rho.flatten()
    if lam is None:
        lam = np.ones((n,m))
    lam_diag = np.diag(np.append(lam.flatten(), lam.flatten()))
    BC = "Neumann"

    #DIFFERENCE SCHEMES
    FD = {'ForwardEuler' : [D2xFE,D2yFE], 'BackwardEuler' : [D2xBE,D2yBE], 'Centered': [D2xCD,D2yCD]}
    Dx = FD[diff][0](n,m, bc = BC, matrix=True)/hx
    Dy = FD[diff][1](n,m, bc = BC, matrix=True)/hy
    Div = np.hstack((Dx.T,Dy.T))
    Grad = np.vstack((Dx,Dy))

    #FIRST STEP
    lam_t = cp.Variable(n*m*2)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Div@lam_diag@lam_t-f)))
    prob.solve()
    lam_t = lam_t.value.reshape(2,n*n)

    #SCALE LAMBDA
    norm = np.array([np.linalg.norm(_lam_t) for _lam_t in lam_t.T])
    lam = norm
    lam = np.diag(np.append(lam,lam))

    lam = HarmAvg(norm.reshape((n,n)))

    lam_diag = np.diag(np.append(lam,lam))

    #PLOTTING
    if vis:
        plt.figure(figsize=(16,10))
        plt.subplot(221)
        plt.quiver(xx,yy,lam_t[0],lam_t[1], color='r')
        plt.title('First Iteration $\\lambda_t$')
        plt.subplot(222)
        plt.imshow(lam_p.reshape(n,n)[::-1])
        plt.title('First Iteration $\\lambda$')
        plt.colorbar()
        plt.subplot(223)
        plt.title('Second Iteration $\\lambda$')
        plt.imshow(lam[::-1])
        plt.subplot(224)
        plt.title('$\\nabla\\phi$')
        plt.quiver(xx,yy,lam_t[0], lam_t[1],color='r')
        plt.show()

    return lam
