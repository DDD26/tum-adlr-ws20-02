### define our robot environment ###
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn

class MPEnv():
    def __init__(self, pos, size, bound, start, end, opt_num = 5, sp_num =5,co_num=10, weight = 0.1, voxel_size = np.array([0.1,0.1]), eps=1, order=1):
        """
        Args: 
            pos: np array with shape [N,2], with N being number of obstacles, indicating coordinate of obstacle'slower left corner
            size: np array with shape [N,2], with N being number of obstacles, indicating width and hight of obstacles
            bound: np array with shape [2,], upper boundary of the work space. Lower bound is implicitly (0,0)
            start: np array with shape [2,], start point coordinate of the robot
            end: np: array with shape [2,], end point coordinate of the robot
            opt_num: number of optimization points 
            sp_num: number of subsampling points on line segements between two optimization points for calculating objective
            co_num: number of subsampling points on line segements between two optimization points for collision check
            weight: weight term for length objective 
            voxel_size: np array with shape [2,], voxel_size in x and y direction
            eps: scalar, parameter in cost function
            order: positive int, interpolation order 
        """
        self.pos = pos.astype(int)
        self.size = size.astype(int)
        self.bound = bound.astype(int)
        self.start = start 
        self.end = end
        self.opt_num = opt_num 
        self.sp_num = sp_num
        self.co_num = co_num
        self.weight = weight  #length weight 
        self.voxel_size = voxel_size
        self.eps = eps
        self.order = order 
       
        self.obstacle = self.obstacle()
        self.dis = self.dis() 
        self.dis_der = self.dis_der()
        self.dis_fun = self.dis_fun1()
        self.dis_der_fun = self.dis_der_fun1()
        self.cost_fun = self.cost_fun1()
        self.cost_der_fun = self.cost_der_fun1()
        self.cost_cost_der_mat()
        self.ob_fun = self.ob_fun1()
        self.ob_der_fun = self.ob_der_fun1()
                
    def obstacle(self):
        """
        Geometric shape of the environment 
        Returns: 
            obstacle: a boolean numpy array with shape [bound[0],bound[1]], True indicates obstacle, False indicates free 
        """
        pos = self.pos
        size = self.size 
        bound = self.bound
        obstacle = np.zeros(bound,dtype = bool)
        for i in range(pos.shape[0]):
            low_left = pos[i]
            up_right = low_left + size[i]
            obstacle[tuple(map(slice, low_left, up_right))] = True
        return obstacle 
    
    def dis(self):
        """
        Create nearest distance field, negative indicates inside obstacle
        Returns: 
            dis: a float numpy array with shape [bound[0],bound[1]]
        """
        bound = self.bound
        voxel_size = self.voxel_size 
        
        im = self.obstacle
            
        pad = np.ones(self.bound+2, dtype=bool)
        pad[1:bound[0]+1,1:bound[1]+1] = im    
            
        dis = ndimage.distance_transform_edt(-pad.astype(int) + 1, sampling=voxel_size)
        dis1 = ndimage.distance_transform_edt(pad.astype(int), sampling=voxel_size)
        dis[pad] = - dis1[pad]  # Add interior information
            
        dis = dis[1:bound[0]+1,1:bound[1]+1]
        return dis
    
    def dis_der(self):
        """
        Applying sobel filter to nearest distance field to get and x and y gradient field 
        Returns: 
            dis_der: a float numpy array with shape [2,bound[0],bound[1]], dis_der[0] x gradient and dis_der[1] y gradient 
        """
        dis_der = np.zeros((2,self.bound[0],self.bound[1]),dtype=np.float32)
        for d in range(2):  # Treat image boundary like obstacle
            dis_der[d, ...] = ndimage.sobel(self.dis, axis=d, mode='constant', cval=0)/self.voxel_size[d]
        return dis_der
    
    def dis_fun1(self):
        """
        Interpolate the nearest distance to get distance function
        Returns: 
            dis_fun: a function whose input is float numpy array with shape [N,2], N is number of inquiry points
                                      output is float numpy array with shape [N,], respecting cost of each inquiry points
        """ 
        factor = 1/self.voxel_size
        im = self.dis
        def dis_fun(x):
            x = np.multiply(x,factor)-0.5
            out = ndimage.map_coordinates(im, coordinates=x.T, order=self.order, mode='nearest')
            return out          
        return dis_fun
    
    def dis_der_fun1(self):
        """
        Interpolate the x and y gradient field to get distance gradient function
        Returns: 
            dis_der_fun: a function whose input is float numpy array with shape [N,2], N is number of inquiry points
                                          output is float numpy array with shape [N,2], respecting x and y gradient of each point
        """ 
        der = self.dis_der
        factor = 1/self.voxel_size
        def dis_der_fun(x):
            x = np.multiply(x,factor)-0.5
            gx = ndimage.map_coordinates(der[0,...], coordinates=x.T, order=self.order, mode='nearest')
            gy = ndimage.map_coordinates(der[1,...], coordinates=x.T, order=self.order, mode='nearest')
            return np.stack((gx,gy),axis=0).T
        return dis_der_fun
    
    def cost_fun1(self):
        """
        Assign cost to nearest distance field
        Returns: 
            cost_fun: a function whose input is float numpy array with shape [N,2], N is number of inquiry points
                                       output is float numpy array with shape [N,], cost of each point
        """
        eps = self.eps
        def cost_fun(x):
            dis = self.dis_fun(x)
            cost = np.zeros(dis.shape,dtype=np.float64)
            cost[dis>eps] = 0
            cost[np.logical_and(dis>0,dis<=eps)] = np.square(dis[np.logical_and(dis>0,dis<=eps)]-eps)/(2*eps)
            cost[dis<=0] = eps/2 - dis[dis<=0]
            return cost
        return cost_fun

    def cost_der_fun1(self):
        """
        Assign cost gradient
        Returns: 
            cost_der_fun: a function whose input is float numpy array with shape [N,2], N is number of inquiry points
                                           output is float numpy array with shape [N,2], x and y cost gradient of each point
        """
        eps = self.eps
        def cost_der_fun(x):
            dis = self.dis_fun(x)
            dis_der = self.dis_der_fun(x)
            der = cost = np.zeros((len(dis),2),dtype=np.float64)
            der[dis>eps] = 0
            der[np.logical_and(dis>0,dis<=eps)] = np.multiply((dis[np.logical_and(dis>0,dis<=eps)]-eps),dis_der[np.logical_and(dis>0,dis<=eps)].T).T/eps
            der[dis<=0] = - dis_der[dis<=0]
            return der 
        return cost_der_fun
    
    def cost_cost_der_mat(self):
        x,y = np.mgrid[0:self.bound[0]:1, 0:self.bound[1]:1]
        positions = np.vstack([x.ravel(), y.ravel()])
        positions = positions.T
        positions = (positions+0.5)*self.voxel_size
        self.cost_mat = self.cost_fun(positions).reshape(self.bound)
        self.cost_der_mat = self.cost_der_fun(positions).reshape(list(self.bound)+[2]).transpose(2,0,1)
        
    def real2pix(self,x):
        x = np.around(np.multiply(x,1/self.voxel_size)-0.5).astype(np.uint8)
        pix = np.clip(x,0,63)
        return pix
    
    def pix2real(self,pix):
        x = (pix+0.5)*self.voxel_size
        return x
    
    def ob_fun1(self):
        """
        Given a trajectory, calculate its obstacle cost and length cost and objective 
        Returns: 
            ob_fun: a function whose input is float numpy array with shape [opt_num, 2]
                                     output is float numpy scalar, the objective value
        """
        w = self.weight
        start = self.pix2real(self.start)
        end = self.pix2real(self.end)
        def ob_fun(x):
            x1 = self.all_points(x,self.sp_num)
            x1 = np.delete(x1,0,0)
            x1 = np.delete(x1,x1.shape[0]-1,0)
            return np.mean(self.cost_fun(x1)) + w*np.sum(np.diff(np.insert(x,(0,x.shape[0]),(start,end),axis=0),axis=0)**2)
        return ob_fun
    
    def ob_der_fun1(self):
        """
        Derivative of objective function
        Returns: 
            ob_der_fun: a function whose input is a float numpy array with shape [opt_num, 2]
                                         output is a float numpy array with shape [opt_num,2], the derivative 
        """
        w = self.weight
        opt_num = self.opt_num
        sp_num = self.sp_num
        start = self.pix2real(self.start)
        end = self.pix2real(self.end)
        def ob_der_fun(x):
            ### gradient of obstacle cost ###
            x1 = self.all_points(x,self.sp_num)
            x1 = np.delete(x1,0,0)
            x1 = np.delete(x1,x1.shape[0]-1,0)
            x1 = self.cost_der_fun(x1)
            x1 = torch.Tensor(x1).reshape(1,1,x1.shape[0],x1.shape[1])
            kernel1 = np.append(np.arange(1,sp_num+2,1),np.arange(sp_num,0,-1))/(sp_num+1)
            kernel1 = torch.Tensor(kernel1).reshape(1,1,kernel1.shape[0],1)
            re1 = torch.nn.functional.conv2d(x1,kernel1,stride=(sp_num+1,1))
            re1 = re1/(opt_num+(opt_num+1)*sp_num)
            re1 = torch.squeeze(torch.squeeze(re1,0),0).numpy()
            ### gradient of length cost ###
            x2 = np.insert(x,(0,x.shape[0]),(start,end),axis=0)
            x2 = torch.Tensor(x2).reshape(1,1,x2.shape[0],x2.shape[1])
            kernel2 = torch.Tensor([-1,2,-1]).reshape(1,1,3,1)
            re2 = 2*w*torch.nn.functional.conv2d(x2,kernel2,stride=1)
            re2 = torch.squeeze(torch.squeeze(re2,0),0).numpy()
            return re1+re2
        return ob_der_fun
    
    def all_points(self,x,num):
        """
        Combine all start, end, optimization and subsampling points (both for calculating objective and collision check)
        Args:
            x: float numpy array with shape [opt_num,2], optimization points
            num: number of subsampling points
        Returns: 
            x1: float numpy array with shape [opt_num+2+num*(opt_num+1), 2]
        """
        start = self.pix2real(self.start)
        end = self.pix2real(self.end)
        x1 = np.insert(x,(0,x.shape[0]),(start,end),axis=0)
        for i in range(x1.shape[0]-1):
            x2 = np.linspace(x1[i+(num)*i],x1[i+1+(num)*i],num+1,endpoint=False)
            x2 = np.delete(x2,0,0)
            x1 = np.insert(x1,i+1+(num)*i,x2,axis=0)
        return x1
    
    def initial(self, random_start = False):
        """
        Initialize the trajectory by connecting start, end point and uniform sampling along this line segment
        Returns:
            x0: float numpy array with shape [opt_num, 2] initial optimization points
        """
        start = self.pix2real(self.start)
        end = self.pix2real(self.end)
        if random_start:
            x0 = np.zeros((self.opt_num, 2))
            low_x, high_x = (1, self.environment.bound[0]-1) * self.environment.voxel_size
            low_y, high_y = (1, self.environment.bound[1]-1) * self.environment.voxel_size
            range_xy = np.array([high_x-low_x, high_y-low_y])
            for i in range(self.opt_num):
                x0[i,:] = np.random.random((1,2)) * range_xy + np.array([low_x, low_y])
            x0.sort(axis = 0)
        else:
            x0 = np.linspace(start, end, self.opt_num + 1, endpoint=False)
            x0 = np.delete(x0, 0, 0)
        return x0

    def collision(self,x):
        """
        Check if any of optimization and subsampling points collides with any of the obstacles. Moreover check if all points are in the boundary. 
        If both conditions are satisfied, returns True (collision free), otherwise False.    
        """
        low = self.pos
        high = self.pos + self.size
        x1 = self.all_points(x,self.co_num)
        x1 = np.multiply(x1,1/self.voxel_size)-0.5
        out = np.empty((x1.shape[0],),dtype=bool)
        for i in range(x1.shape[0]):
            k = np.concatenate((x1[i]>low,x1[i]<high),axis=1)
            k = np.all(k,axis=1)
            out[i] = np.any(k)
            out1 = np.any(out)
        out2 = np.all([x1>0,x1<self.bound])
        return not out1 and out2