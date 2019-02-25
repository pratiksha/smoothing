import subprocess

def main():
    nmodels = 4
    noise = [0.25, 0.5]

    for noise_sd in noise:
        for idx in range(nmodels):
            train_cmd = f"python code/train.py cifar10 cifar_resnet110 lowres_models/cifar10/resnet110/noise_{noise_sd}/{idx} --subsample --index {idx} --noise_sd {noise_sd} --gpu 0".format(noise_sd=noise_sd, idx=idx)
            certify_cmd = f"python code/certify.py cifar10 lowres_models/cifar10/resnet110/noise_{noise_sd}/{idx}/checkpoint.pth.tar {noise_sd} lowres_data/certify/cifar10/resnet110/noise_{noise_sd}/{idx} --skip 20 --batch 1000 --subsample".format(noise_sd=noise_sd, idx=idx)

            print(train_cmd)
            subprocess.call(train_cmd, shell=True)
            print(certify_cmd)
            subprocess.call(certify_cmd, shell=True)

if __name__=='__main__':
    main()
