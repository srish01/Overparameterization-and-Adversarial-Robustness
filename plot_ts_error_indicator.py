import pickle
import argparse
import logging

from matplotlib import pyplot as plt

from config import REG_TINYNET_PRETRAINED_MNIST, REG_TINYNET_PRETRAINED_CIFAR10, PLOT_FOLDER, MNIST_FCRELU_INDICATOR_PLOT, MNIST_CNN_INDICATOR_PLOT, CIFAR10_CNN_INDICATOR_PLOT, FCRELU_PRETRAINED_CIFAR10, FCRELU_PRETRAINED_MNIST, CIFAR10_FCRELU_INDICATOR_PLOT

failure_indicator = {
    "incomplete_optimization": "Non-conververgance Issue ",
    "unavailable_gradients": "Shattered Gradients Issue",
    "unstable_predictions": "Stochastic Gradients Issue",
    "silent_success": "Attack Implementation Issue",
    "transfer_failure": "Non-adaptation Issue",
    "unconstrained_attack_failure": "Unreachable Misclassification Issue",
    "attack_success": "Attack Success"
}

logging.basicConfig(level=logging.INFO, filename='plot.log', filemode='w')

def plot_robustness(results, ds, model):
    epsilons = results['epsilons']
    plt.figure()
    for p in results['overparam']:
        print(f"Params: {p}")
        robacc = [results['overparam'][p]['robacc'][eps] for eps in epsilons]
        print(robacc)
        plt.plot(epsilons, robacc, label=f'p{p}')
    plt.xlabel("Epsilon")
    plt.ylabel("Robust accuracy")
    plt.legend()
    plt.savefig(str(PLOT_FOLDER / f"{ds}_{model}_robustmess_plot.jpg"))
    # plt.show()


def plot_indicator(results, key, indicator_name, title):
    epsilons = results['epsilons']
    for p in results:
        print(f"Params: {p}")
        ind = results['overparam'][p]['indicators']
        ind_values = []
        for eps in epsilons:
            eps_ind = ind[eps]
            failed = eps_ind[eps_ind['attack_success'] == False]
            failed_attacks = failed[key].mean() if failed.size > 0 else 0
            ind_values.append(failed_attacks)
        plt.plot(epsilons, ind_values, label=f"{indicator_name} - {p}")
    plt.xlabel("Epsilon")
    plt.ylabel("Indicator")
    plt.title(title)
    plt.legend()
    plt.show()


def barplot_indicators(results, ds, model, key):
    logging.info(f"\n\n\t\t Analysis: {key}\t\n")
    epsilons = results['epsilons']
    issue = failure_indicator[key]
    for p in results['overparam']:
        print(f"Plot barplot for params {p}")
        fname = f"param_{p}_{key}.jpg"
        if ds == 'mnist': 
            if model == "cnn":
                fpath = MNIST_CNN_INDICATOR_PLOT / fname
            elif model == "2fcRelu":
                fpath = MNIST_FCRELU_INDICATOR_PLOT / fname
        elif ds == 'cifar10':
            if model == 'cnn':
                fpath = CIFAR10_CNN_INDICATOR_PLOT / fname
            elif model == '2fcRelu':
                fpath = CIFAR10_FCRELU_INDICATOR_PLOT / fname
        ind = results['overparam'][p]['indicators']
        ind_values = []
        plt.figure()
        for eps in epsilons:
            eps_ind = ind[eps]
            failed = eps_ind[eps_ind['attack_success'] == False]
            # print(f"for eps: {eps}, failed: \n{failed}")
            succeeded = eps_ind[eps_ind['attack_success'] == True]
            if key == "attack_success":
                logging.info(f"--For dataset: {ds}, parameter: {p}, eps: {eps}")
                logging.info(f"Total Num of failed attacks: {str(len(failed))}")
                logging.info(f"Total Num of succeeded attacks: {str(len(succeeded))}")
            if not key == "attack_success":
                # if failed.size > 0:
                #     failed_attacks = failed[key].mean() 
                #     if key in ["unavailable_gradients", "silent_success", "transfer_failure"]:
                #         num_failed_attacks = sum(failed[key])
                #     elif key in ["incomplete_optimization", "unconstrained_attack_failure"]:
                #         num_failed_attacks = sum(failed[key])  
                #     elif key == "unstable_predictions":
                #         num_failed_attacks = sum(failed[key] > 1)
                # else:
                #     0
                # print(f"failed attacks due to {key}: \n{num_failed_attacks}")
                failed_attacks = failed[key].mean() if failed.size > 0 else 0
                ind_values.append(failed_attacks)
                
            # logging.info(f"From [{p}: {eps}], {len(failed)} attacks failed")
            # logging.info(f"{num_failed_attacks} attacks failed due to '{key}'")
        if not key == "attack_success":
            plt.bar(epsilons, ind_values)
            plt.ylim([0, 1])
            plt.title(f"{issue} on {p} params")
            plt.savefig(fpath, bbox_inches="tight")
        # plt.show()


if __name__ == "__main__":
    
    # logger = logging.getLogger(__name__)
    # f_handler = logging.FileHandler('plot.log')
    # f_handler.setLevel(logging.INFO)
    # logger.addHandler(logging.Formatter('%(asctime)s - %(message)s'))

    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', dest='dataset', choices=['mnist', 'cifar10'])
    parser.add_argument('-model', dest = 'model', choices = ['cnn', '2fcRelu'])
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model

    if dataset == 'mnist':
        if model == 'cnn':
            indicators_path = REG_TINYNET_PRETRAINED_MNIST / "indicators.pkl"
            logging.info(f"***** MNIST - CNN *****")
        elif model == '2fcRelu':
            indicators_path = FCRELU_PRETRAINED_MNIST / "indicators.pkl"
    elif dataset == 'cifar10':
        if model == 'cnn':
            indicators_path = REG_TINYNET_PRETRAINED_CIFAR10 / "indicators.pkl"
            logging.info(f"***** CIFAR10 *****")
        elif model == '2fcRelu':
            indicators_path = FCRELU_PRETRAINED_CIFAR10 / "indicators.pkl"

    with open(str(indicators_path), 'rb') as h:
        results = pickle.load(h)

    plot_robustness(results, dataset, model)
    barplot_indicators(results, dataset, model, 'attack_success')
    barplot_indicators(results, dataset, model, 'incomplete_optimization')
    barplot_indicators(results, dataset, model, 'unavailable_gradients')
    barplot_indicators(results, dataset, model, 'unstable_predictions')
    barplot_indicators(results, dataset, model, 'silent_success')
    barplot_indicators(results, dataset, model, 'transfer_failure')
    barplot_indicators(results, dataset, model, 'unconstrained_attack_failure')


