import time
# import pprint
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from model import *


def attention_pandas_seaborn():

    with open('reverse_index.pickle', 'rb') as f:
        reverse_index = pickle.load(f)

    print('Building graph...')

    dataset = tf.placeholder_with_default(2, [])

    document_batch, document_weights,\
        query_batch, query_weights, answer_batch = read_records(dataset)

    y_hat, reg, M, alpha, beta, query_importance, attentions = inference(
        document_batch, document_weights, query_batch, query_weights)

    loss, train_op, global_step, accuracy = train(
        y_hat, reg, document_batch, document_weights, answer_batch)

    saver = tf.train.Saver()

    print('Starting sesion...')
    with tf.Session() as sess:

        sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()])

        model = tf.train.latest_checkpoint(model_path)
        if model:
            print('Restoring ' + model)
            saver.restore(sess, model)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print('hopefully prints something')

        start_time = time.time()

        print(time.strftime('%X %x'))

        try:

            acc, d, dm, q, qm, M, a, b, qa, da, ans, pred = sess.run(
                [accuracy,
                 document_batch,
                 document_weights,
                 query_batch,
                 query_weights,
                 M,
                 alpha,
                 beta,
                 query_importance,
                 attentions,
                 answer_batch,
                 y_hat],
                feed_dict={dataset: 2})

            pred = tf.argmax(pred, 1).eval()
            print('pred', pred)

            elapsed_time, start_time = time.time() - start_time, time.time()

            print(time.strftime('%X %x'))
            print(acc, elapsed_time)

            for v in [acc, d, dm, q, qm, M, a, b, qa, da]:
                print(v.shape)
            print()

        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()

        coord.join(threads)

        # pp = pprint.PrettyPrinter(indent=4)

        for i in range(FLAGS.batch_size):
            # for i in range(3):

            # print(q[i].flatten().shape, qm[i].shape)
            # pp.pprint(list(zip(q[i].flatten(), qm[i])))
            # print(q[i, :np.count_nonzero(qm[i])])

            # pp.pprint(list(zip(att[i], dm[i])))

            # image = a[i, :np.count_nonzero(dm[i]), :np.count_nonzero(qm[i])]
            # doc_att = np.expand_dims(da[i, :np.count_nonzero(dm[i])], axis=1)

            len_doc = np.count_nonzero(dm[i])
            len_query = np.count_nonzero(qm[i])

            matrix = M[i, :len_doc, :len_query]
            alpha = a[i, :len_doc, :len_query]
            beta = b[i, :len_doc, :len_query]
            qry_att = qa[i, :len_query].reshape(1, -1)
            doc_att = np.expand_dims(da[i, :len_doc], axis=1)

            document = d[i, :len_doc]
            question = q[i, :len_query]

            d_words = [reverse_index[i] for i in document]
            q_words = [reverse_index[i] for i in question]
            answer = reverse_index[ans[i]]
            prediction = reverse_index[pred[i]]

            print('Doc:', d_words[:10])
            print('Qry:', q_words)
            print('Ans:', answer)
            print('Pred:', prediction)

            data = {'m': matrix,
                    'a': alpha,
                    'b': beta,
                    'q': qry_att,
                    'd': doc_att,
                    'dw': document,
                    'qw': question,
                    }

            for k, v in data.items():
                print(k, v.shape)

            m_df = pd.DataFrame(matrix, index=d_words, columns=q_words)
            a_df = pd.DataFrame(alpha, index=d_words, columns=q_words)
            b_df = pd.DataFrame(beta, index=d_words, columns=q_words)
            qry_df = pd.DataFrame(
                qry_att, index=['attention'], columns=q_words)
            doc_df = pd.DataFrame(doc_att, index=d_words,
                                  columns=['attention'])

            figures = {'m': m_df,
                       'a': a_df,
                       'b': b_df,
                       'q': qry_df,
                       'd': doc_df}

            # titles = {'m': 'MATRIX',
            #           'a': 'ALPHA',
            #           'b': 'BETA',
            #           'q': 'QUERY',
            #           'd': 'DOCUMENT'}

            # print(m_df.head())
            # print(a_df.head())
            # print(b_df.head())

            # os.makedirs('plots/'+str(i+1), exist_ok=True)

            gs = gridspec.GridSpec(1, 5)
            # Draw a heatmap with the numeric values in each cell
            # sns.set()
            fs = (question.shape[0] / 2 * 5, document.shape[0] / 4)
            plt.figure(figsize=fs)
            plt.subplots_adjust(top=0.9)

            for sub_fig, name in enumerate(['m', 'a', 'b', 'q', 'd']):
                data = figures[name]
                print(str(i + 1) + ' ' + name)

                sns.set()
                # fs = (question.shape[0] / 2, document.shape[0] / 4)
                # f, ax = plt.subplots(figsize=fs)
                sub = plt.subplot(gs[sub_fig])
                if name == 'm':
                    sns.heatmap(
                        data,
                        square=True,
                        label='small',
                        # annot=True,
                        fmt="d",
                        center=0,
                        # linewidths=.5,
                        # vmin=0, vmax=1,
                        # cbar=False,
                        # ax=ax,
                    )
                else:
                    sns.heatmap(
                        data,
                        square=True,
                        label='small',
                        # annot=True,
                        fmt="d",
                        # center=0,
                        # linewidths=.5,
                        # vmin=0, vmax=1,
                        # cbar=False,
                        # ax=ax,
                    )

                # sub.set_title(titles[name], fontsize=24, y=1.005)
                sub.axes.xaxis.tick_top()
                sub.set_xticklabels(sub.get_xticklabels(), rotation=90)
                # plt.title(
                #   'Doc: '+ str(len_doc) +\
                #     ' Query: '+ str(len_query) +\
                #     ' ANS: '+ answer,
                #     ' PRED: '+ prediciton,
                #   y=1.05)

            plt.suptitle(
                'Doc: ' + str(len_doc) +
                ' Query: ' + str(len_query) +
                ' ANS: ' + answer +
                ' PRED: ' + prediction,
                y=1.01,
                fontsize=48
            )
            # x1,x2,y1,y2 = plt.axis()
            # plt.axis((x1, x2, y1, y2 + 1000))
            plt.tight_layout()
            # plt.savefig('plots/'+str(i+1)+'/'+name+'.png')
            # plt.savefig('plots/'+str(i+1)+'/all.jpg')
            plt.savefig('plots/' + str(i + 1) + '.png')
            plt.close()

            with open('plots/' + str(i + 1) + '.txt', 'w') as texts:
                texts.write('Pred: ' + prediction + '\n')
                texts.write('Ans: ' + prediction + '\n\n')
                texts.write(' '.join(q_words) + '\n\n')
                texts.write(' '.join(d_words) + '\n\n')

            print()


if __name__ == "__main__":
    attention_pandas_seaborn()
