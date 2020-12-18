# Term Frequency - Inverse Document Frequency tf-idf

## Definition of tf-idf

We are using the tf-idf definition of [scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):

tf-idf is the multiplication of  tf (term-frequency) with idf (inverse document frequency):
$$
\begin{equation}
{\rm tf\!\!-\!\!idf}(t,d) = {\rm tf}(t,d)\cdot{\rm idf}(t)
\end{equation}
$$
The term frequency tf (t) of a term $t$ is defined as the number of times a term occurs in a given document.

The inverse document frequency idf (t) of a term $t$ is defined as:

$$
{\rm idf}(t) = log\left(\frac{1+n}{1+{\rm df}(t)}+1\right)
$$

$n$ is the total number of documents in the document set,

df (t) is the number of documents in the document set that contain term $t$.

The resulting tf-idf vectors are normalized by the Euclidean norm:
$$
v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v_1^2+v_2^2+ ... +v_n^2}}
$$


