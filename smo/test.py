import svmMLiA
import time
if __name__ == "__main__":
    start = time.time()
    kernel =['lin', 'rbf']
    alphas_all, b_all = svmMLiA.train_step(kTup=('rbf', 10))

    svmMLiA.predict_digits(alphas_all, b_all, kTup=('rbf', 10))
    end = time.time()
    print("All done in %.2f seconds" % (end - start))
