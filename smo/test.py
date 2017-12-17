import svmMLiA
import time
import pickle

if __name__ == "__main__":
    start = time.time()
    kernel = ['lin', 'rbf']
    alphas_all, b_all = svmMLiA.train_step(kTup=('rbf', 10))

    # saving parameter
    with open('alpha_b.pkl', 'wb') as f:
        pickle.dump([alphas_all, b_all], f)

    # Getting back the objects:
    with open('alpha_b.pkl', 'rb') as f:
        alphas_all, b_all = pickle.load(f)

    svmMLiA.predict_digits(alphas_all, b_all, kTup=('rbf', 10))
    end = time.time()
    print("All done in %.2f seconds" % (end - start))
