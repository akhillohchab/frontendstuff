import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy
import pickle

class pmf_func():

    def __init__(self, rating_tup, latent=1):
        self.latent = latent
        self.learn_rate = .0001
        self.reg_str = 0.1
        
        self.ratings = numpy.array(rating_tup).astype(float)
        self.converged = False

        self.n_users = int(numpy.max(self.ratings[:, 0]) + 1)
        self.n_items = int(numpy.max(self.ratings[:, 1]) + 1)
        
        print (self.n_users, self.n_items, self.latent)
        print self.ratings

        self.users = numpy.random.random((self.n_users, self.latent))
        self.items = numpy.random.random((self.n_items, self.latent))

        self.new_users = numpy.random.random((self.n_users, self.latent))
        self.new_items = numpy.random.random((self.n_items, self.latent))           


    def likely(self, users=None, items=None):
        if users is None:
            users = self.users
        if items is None:
            items = self.items
            
        sq_error = 0
        
        for rat_tup in self.ratings:
            if len(rat_tup) == 3:
                (i, j, rating) = rat_tup
                weight = 1
            elif len(rat_tup) == 4:
                (i, j, rating, weight) = rat_tup
            
            r_hat = numpy.sum(users[i] * items[j])

            sq_error += weight * (rating - r_hat)**2

        L2_norm = 0
        for i in range(self.n_users):
            for d in range(self.latent):
                L2_norm += users[i, d]**2

        for i in range(self.n_items):
            for d in range(self.latent):
                L2_norm += items[i, d]**2

        return -sq_error - self.reg_str * L2_norm
        
        
    def update(self):

        old_updates = numpy.zeros((self.n_users, self.latent))
        d_updates = numpy.zeros((self.n_items, self.latent))        

        for rat_tup in self.ratings:
            if len(rat_tup) == 3:
                (i, j, rating) = rat_tup
                weight = 1
            elif len(rat_tup) == 4:
                (i, j, rating, weight) = rat_tup
            
            r_hat = numpy.sum(self.users[i] * self.items[j])
            
            for d in range(self.latent):
                old_updates[i, d] += self.items[j, d] * (rating - r_hat) * weight
                d_updates[j, d] += self.users[i, d] * (rating - r_hat) * weight

        while (not self.converged):
            initial_lik = self.likely()

            print "  setting learning rate =", self.learn_rate
            self.updates_try(old_updates, d_updates)

            final_lik = self.likely(self.new_users, self.new_items)

            if final_lik > initial_lik:
                self.updates_apply(old_updates, d_updates)
                self.learn_rate *= 1.25

                if final_lik - initial_lik < .1:
                    self.converged = True
                    
                break
            else:
                self.learn_rate *= .5
                self.updates_undo()

            if self.learn_rate < 1e-10:
                self.converged = True

        return not self.converged
    

    def updates_apply(self, old_updates, d_updates):
        for i in range(self.n_users):
            for d in range(self.latent):
                self.users[i, d] = self.new_users[i, d]

        for i in range(self.n_items):
            for d in range(self.latent):
                self.items[i, d] = self.new_items[i, d]                

    
    def updates_try(self, old_updates, d_updates):        
        alpha = self.learn_rate
        beta = -self.reg_str

        for i in range(self.n_users):
            for d in range(self.latent):
                self.new_users[i,d] = self.users[i, d] + \
                                       alpha * (beta * self.users[i, d] + old_updates[i, d])
        for i in range(self.n_items):
            for d in range(self.latent):
                self.new_items[i, d] = self.items[i, d] + \
                                       alpha * (beta * self.items[i, d] + d_updates[i, d])
        

    def updates_undo(self):
        pass


    def print_lat_v(self):
        print "Users"
        for i in range(self.n_users):
            print i,
            for d in range(self.latent):
                print self.users[i, d],
            print
            
        print "Items"
        for i in range(self.n_items):
            print i,
            for d in range(self.latent):
                print self.items[i, d],
            print    


    def lat_v_save(self, prefix):
        self.users.dump(prefix + "%sd_users.pickle" % self.latent)
        self.items.dump(prefix + "%sd_items.pickle" % self.latent)
    

def generated_ratings(noise=.25):
    u = []
    v = []
    ratings = []

    my_file = open("C:\Users\Akhil\Downloads\ml-100k\dataset.csv", "rb")
    i=0
    j=0
    ratings=[]
    for line in my_file:
        i=i+1
        l = [k.strip() for k in line.split(',')]
        #print l,len(l)
        ratings.append((l[0],l[1],l[2]))
        #print ratings[j]
        j=j+1
    #print ratings

    print "ratings stored"    
    return ratings


def plotratings(ratings):
    xs = []
    ys = []
    
    for i in range(len(ratings)):
        xs.append(ratings[i][1])
        ys.append(ratings[i][2])
    
    pylab.plot(xs, ys, 'bx')
    pylab.show()


def plotlvectors(U, V):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    cmap = cm.jet
    ax.imshow(U, cmap=cmap, interpolation='nearest')
    plt.title("Users")
    plt.axis("off")

    ax = fig.add_subplot(122)
    ax.imshow(V, cmap=cmap, interpolation='nearest')
    plt.title("Items")
    plt.axis("off")

def plotprediction(U, V):
    r_hats = -5 * numpy.ones((U.shape[0] + U.shape[1] + 1, 
                              V.shape[0] + V.shape[1] + 1))

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            r_hats[i + V.shape[1] + 1, j] = U[i, j]

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            r_hats[j, i + U.shape[1] + 1] = V[i, j]

    for i in range(U.shape[0]):
        for j in range(V.shape[0]):
            r_hats[i + U.shape[1] + 1, j + V.shape[1] + 1] = numpy.dot(U[i], V[j]) / 10

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(r_hats, cmap=cm.gray, interpolation='nearest')
    plt.title("Predicted Ratings")
    plt.axis("off")


if __name__ == "__main__":

    ratings = generated_ratings()
    #(ratings, true_o, true_d) = generated_ratings()

    #plotratings(ratings)

    pmf = pmf_func(ratings, latent=5)
    
    liks = []
    while (pmf.update()):
        lik = pmf.likely()
        liks.append(lik)
        print "L=", lik
        pass
    
    plt.figure()
    plt.plot(liks)
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")
    
    plotlvectors(pmf.users, pmf.items)
    plotprediction(pmf.users, pmf.items)
    plt.show()

    pmf.print_lat_v()
    pmf.lat_v_save("models/")
