def standardscaler(x_train): 
    mean = np.mean(x_train, axis = 0, keepdims= True)
    std = np.std(x_train, axis = 0, keepdims = True)
    return (x_train - mean)


def covariance_matrix(n, standard_x_train):
    covariance_matrix =  (standard_x_train @ standard_x_train.T)/(n -1) 
    return covariance_matrix

# you alsouse that function  or 


x_train = x_train - np.mean(x_train, axis = 0 )
covariance = np.cov(x_train , rowvar = False )
eignvalues , eignvectors = np.linalg.eig(covariance)

sort_index = np.argsort(eignvalues)[::-1]
eignvalues_sorted = eignvalues[sort_index]
eignvectors_sorted = eignvectors[:,sort_index]

variance_explain = eignvalues_sorted / np.sum(eignvalues_sorted) 
def find_k(variance_explain):
    variance = 0 
    for index , values in enumerate(variance_explain): 
        variance += values 
        if variance >= 0.95 :
            return index + 1

k = find_k(variance_explain)
top_eignvectors = eignvectors_sorted[:k]

pca_componets = x_train @ top_eignvectors
print(pca_componets)


class autoencoder:
    def __init__(self, input_dim, hidden_dim,learning_rate):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim 
        self.learning_rate = learning_rate 

        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01 
        self.b1 = np.zeros((1,hidden_dim))
        self.w2 = np.random.randn(hidden_dim , input_dim ) * 0.01 
        self.b2 = np.zeros((1,input_dim))
    
    def sigmoid(self,x):
        return  1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def forward(self,x):
        self.x = x

        self.z1 = np.dot(x,self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1 , self.w2) + self.b2 
        self.a2 = self.sigmoid(self.z2)

        return self.a2 
    
    def compute_loss(self, y, y_pre):
        return np.mean((y - y_pre)**2 )
    

    def backward(self):
        batch = self.x.shape[0]

        loss = self.a2  - self.x 
        self.d_z2 = loss * self.sigmoid_derivative(self.z2)
        self.d_w2 = (1/batch) * np.dot(self.a1.T,self.d_z2)
        self.d_b2 = (1/ batch) * np.sum(self.d_z2, axis = 0, keepdims = True)

        self.d_z1 = np.dot(self.d_z2, self.d_w2.T) * self.sigmoid_derivative(self.z1)
        self.d_w1 = ( 1/batch) * np.dot(self.x.T, self.d_z1)
        self.d_b1 = (1/ batch) * np.sum(self.d_z1 , axis = 0, keepdims = True)

        self.w2  -= self.learning_rate * self.d_w2 
        self.b2 -= self.learning_rate * self.d_b2
        self.w1 -= self.learning_rate * self.d_w1 
        self.b1 -= self.learning_rate * self.d_b1

    def train(self, x , epochs):
        for i in range(epochs):
            y = self.forward(x)
            loss = self.compute_loss(x,y)
            self.backward()

            if i %  1000 == 0 : 
                print(f"epochs - {epochs}  loss -- {loss}")
    
    def reconstruct(self, x):
        return self.forward(x)
    
    def detect_anomalies(self, x, threshold):
        y_predict= self.reconstruct(x)
        loss = np.mean((x - y_predict)**2, axis = 0)
        return loss > threshold
    


input_dims = pca_componets.shape[1]
hidden_dims = int(input_dims / 2)
learning_rate = 0.001

auto_e = autoencoder(input_dims, hidden_dims,learning_rate)
auto_e.train(pca_componets, epochs=5000)

threshold = 0.1 

anomalies = auto_e.detect_anomalies(pca_componets, threshold) 

print(anomalies)
     
