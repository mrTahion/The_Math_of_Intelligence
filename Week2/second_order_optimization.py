import  numpy as  np
import  pandas  as  pd

cols  = ['BodyweightKg', 'TotalKg', 'Sex', 'Equipment']
dataset = pd.read_csv('openpowerlifting.csv',  skipinitialspace=True, usecols=cols)
dataset = dataset.dropna()

dataset = dataset.loc[(dataset['Sex'] == 'M') & (dataset['Equipment'] == 'Raw')]
dataset = dataset.drop(columns = ['Sex', 'Equipment'] )

data = dataset.as_matrix()


def total_error(point_m_b):
    totalError = 0
    for i in range(0, len(point_m_b)):
        x = data[i, 0]
        y = data[i, 1]
        totalError += (y - (point_m_b[0] * x + point_m_b[1])) ** 2
    return totalError / float(len(point_m_b))



def jacobian(point_m_b, h=5e-6):
    n = len(point_m_b)
    jacobian_matrix = np.zeros(n) 
    for i in range(n):
        x_i = np.zeros(n)
        x_i[i] += h 
        jacobian_matrix[i] = (total_error(point_m_b + x_i) - total_error(point_m_b)) / h
	
	return jacobian_matrix

def hessian(point_m_b, h=5e-6):
    n = len(point_m_b)
    hessian_matrix = np.zeros(n) 
    for i in range(n):
        x_i = np.zeros(n)
        x_i[i] += h 
        hessian_matrix[i] = (jacobian(point_m_b + x_i) - jacobian(point_m_b)) / h

    return hessian_matrix


def newtons_method(init_point_m_b, max_iterations = 5e3): 
    point_m_barr = np.zeros((max_iter, len(init_point_m_b))) 
    best_re = None 
    for i in range(max_iterations):
        jacobian = jacobian(point_m_b_arr[i])
        hessian = hessian(point_m_b_arr[i]) 
        point_m_b_arr[i+1] = point_m_b_arr[i] - np.dot(np.linalg.pinv(hessian), jacobian) 
        
        best_re = point_m_b_arr[i+1]

    return best_re

def run():
    
    init_points_m_b = np.array([0.0, 8.0])
	predict_points = newtons_method(init_points)



if __name__ == '__main__':
    run()


