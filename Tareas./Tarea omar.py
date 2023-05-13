import numpy as np
import matplotlib.pyplot as plt

# El modelo de regression logistica asume que la relacion entre variables esta mediada por la funcion logistica

epsi=np.random.normal(0, 0.1, 100) ### agregamos alguna perturbacion normal


def y_hat(theta_0, theta_1, x):
    p = (theta_0*x) / (theta_1 +x) - epsi
    return(p)

# Simulemos 100 datos para nuestra regression logistica
n = 100

# Definamos los parametros theta_0 "intercepto' y theta_1 "pendiente"
theta_0, theta_1 = [-10, 10]

# Simular los datos x1, y
# Grilla de valores para nuestra covariable
x1 = np.linspace(0, 1, n)


y = np.random.normal(y_hat(theta_0, theta_1, x1), 0.1, n)

# Graficamos
plt.scatter(x1, y, s= 5)
plt.plot(x1, y_hat(0,1, x1), color='red', linewidth = 1) 

# Ahora vamos a realizar un SGD para estimar los parametros theta.
# Inicializar parametros
t0 = 0
t1 = 1

plt.scatter(x1, y, s= 5)
plt.plot(x1, y_hat(-10, 10, x1), color='red', linewidth = .9)
# La curva en verde es la curva estimada con los parametros inicializados (no aprendidos aun)
plt.plot(x1, y_hat(t0, t1, x1), color= 'green', linewidth= 1, linestyle= '--')

# Definimos nuestra Loss function
def loss_fun(Y, X, t0, t1):
    loss = (Y-y_hat(t0,t1,X))*(Y-y_hat(t0,t1,X))
    return(loss)

# Obtenemos la cantidad error que cometemos
loss_ini = loss_fun(y, x1, t0, t1)
print("LOSS inicial:", loss_ini)

# Llevamos regsitro del error y los parametros
error = []
error.append(loss_ini)

t0_hat = []
t1_hat = []

t0_hat.append(t0)
t1_hat.append(t1)

# Gradient descent
def delta_theta_0(y, x1, t0, t1):
    delta_0 = -y*(x1/(t1+x1))-(x1/(t1+x1))*y+2*x1*(x1/(t1+x1))
    return(delta_0)

def delta_theta_1(y, x1, t0, t1):
    delta_1 = y*(t0*x1)+(t0*x1)*y-(t0*x1/(t1+x1))*(t1*x1)
    return(delta_1)

# Step size o learnig rate
rho = .01 # punto uno de la tarea, notamos que si aumentamos el step size
            # en "Learning de los parametros theta" no tiene tanta "embergadura" al comienzo

t0 -= rho * delta_theta_0(y, x1, t0, t1)
t1 -= rho * delta_theta_1(y, x1, t0, t1)

t0_hat.append(t0)
t1_hat.append(t1)

loss = loss_fun(y, x1, t0, t1)
error.append(loss)


print(t0_hat, t1_hat, error, sep='\n')

epoch = 5000

for i in range(epoch):
    t0 -= rho * delta_theta_0(y, x1, t0, t1)
    t1 -= rho * delta_theta_1(y, x1, t0, t1)
    
    t0_hat.append(t0)
    t1_hat.append(t1)
    
    loss = loss_fun(y, x1, t0, t1)
    error.append(loss)

plt.plot(error)
plt.title("Loss en funcion del Epoch")

plt.plot(t0_hat, label = 'theta_0')
plt.axhline(y = theta_0, color='red', linestyle='--')

plt.plot(t1_hat, label = 'theta_1')
plt.axhline(y = theta_1, color='red', linestyle='--')

plt.title("Learning de los parametros theta")


plt.scatter(x1, y, s= 5)
plt.plot(x1, y_hat(-10,10, x1), color='red', linewidth = 1)

for i in range(50):
    plt.plot(x1, y_hat(t0_hat[10*i], t1_hat[10*i], x1), color= 'gray', linewidth= .5, linestyle= '--')

plt.plot(t0_hat[0:6], t1_hat[0:6])

plt.title("Descent de los parametros")
plt.xlabel('theta_0')
plt.ylabel('theta_1')

