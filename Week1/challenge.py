import  numpy as  np
import  pandas  as  pd  #To read the dataset#To rea 
import matplotlib.pyplot as plt #Plotting

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]

def run():
    # Using kaggle openpowerlifting database
	cols  = ['BodyweightKg', 'TotalKg', 'Sex', 'Equipment']
	dataset = pd.read_csv('data/openpowerlifting.csv',  skipinitialspace=True, usecols=cols)
	dataset = dataset.dropna()

	dataset = dataset.loc[(dataset['Sex'] == 'M') & (dataset['Equipment'] == 'Raw')]
	dataset = dataset.drop(columns = ['Sex', 'Equipment'] )

	data = dataset.as_matrix()


	learning_rate=0.0001
	initial_b = 8 # initial y-intercept guess
	initial_m = 0 # initial slope guess
	num_iterations = 1000

	print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, data))
	print "Running..."
	[b, m] = gradient_descent_runner(data, initial_b, initial_m, learning_rate, num_iterations)
	print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, data))

	x=data[:,0]

	plt.figure(figsize=(10,5))
	plt.title('Powerlifter bodyweight vs Best Total')
	plt.scatter(x=x, y=data[:,1])
	plt.plot(x, m*x + b, color='red',label='Fitting Line')
	plt.xlabel('Bodyweight, Kg')
	plt.ylabel('Total, Kg')

	plt.show()

if __name__ == '__main__':
    run()


