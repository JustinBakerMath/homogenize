from homogenize.pdeint import *
import pytest

@pytest.fixture
def domain_21x21():
	#GRID
	n,m=21,21
	y=np.linspace(-1,1,n)
	x=np.linspace(-1,1,m)
	dx=2/(n+1)
	dy=2/(m+1) 
	domain_len=n*m
	domain_shape = (n,m)
	xx,yy=np.meshgrid(x,y)
	fdf = FiniteDifferenceFunctions(m,n,dx,dy)
	fdm = FiniteDifferenceMatrices(m,n,dx,dy)
	return n,m,y,x,dx,dy,domain_len,domain_shape

def test1(domain_21x21):
	actual, _, _, _, _, _, _, _  = domain_21x21
	expected = 21
	assert (actual==expected) 
