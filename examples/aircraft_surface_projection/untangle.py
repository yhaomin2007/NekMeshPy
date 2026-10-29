"""Two-dimensional positive-corner-Jacobian smoothing with fixed CAD boundaries."""
import numpy as np
from scipy.optimize import minimize
def smooth(start,quads,fixed,sign,maxiter=800):
 P=start.copy();free=np.flatnonzero(~fixed);reference=P.copy();corners=np.stack([quads,np.roll(quads,-1,axis=1),np.roll(quads,1,axis=1)],axis=2).reshape(-1,3);i,j,k=corners.T
 U=P[j]-P[i];V=P[k]-P[i];scale=np.maximum(np.linalg.norm(U,axis=1)*np.linalg.norm(V,axis=1),1e-8)
 def state(P):
  U=P[j]-P[i];V=P[k]-P[i];d=sign*(U[:,0]*V[:,1]-U[:,1]*V[:,0]);return U,V,d
 for eps in [.1,.025,.005,.001,.0002]:
  epsilon=eps*scale
  def objective(x):
   P[free]=x.reshape(-1,2);U,V,d=state(P);s=(U*U+V*V).sum(1);root=np.sqrt(d*d+4*epsilon*epsilon);den=np.where(d>=0,d+root,4*epsilon*epsilon/(root-d));f=s/den;gu=2*U/den[:,None]-(f/root)[:,None]*sign*np.c_[V[:,1],-V[:,0]];gv=2*V/den[:,None]-(f/root)[:,None]*sign*np.c_[-U[:,1],U[:,0]];G=np.zeros_like(P);np.add.at(G,j,gu);np.add.at(G,k,gv);np.add.at(G,i,-gu-gv)
   anchor=.001*np.sum((P[free]-reference[free])**2);G[free]+=.002*(P[free]-reference[free]);return f.mean()+anchor,G[free].ravel()/len(f)+.002*(P[free]-reference[free]).ravel()*(1-1/len(f))
  result=minimize(objective,P[free].ravel(),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-8,'maxls':40,'maxcor':20});P[free]=result.x.reshape(-1,2);d=state(P)[2];print('surface-chart smoothing',eps,'bad corners',sum(d<=0),'minimum',d.min(),'iterations',result.nit,flush=True)
  if np.all(d>1e-12) and eps<=.005:break
 return P
