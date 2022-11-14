# SYDE 556/750 --- Assignment 4
**Student ID: 00000000**

*Note:* Please include your numerical student ID only, do *not* include your name.

*Note:* Refer to the [PDF](https://github.com/celiasmith/syde556-f22/raw/master/assignments/assignment_04/syde556_assignment_04.pdf) for the full instructions (including some hints), this notebook contains abbreviated instructions only. Cells you need to fill out are marked with a "writing hand" symbol. Of course, you can add new cells in between the instructions, but please leave the instructions intact to facilitate marking.


```python
# Import numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Import nengo and some helper functions for Q1
import nengo
from nengo.utils.ensemble import tuning_curves
from nengo.utils.connection import eval_point_decoding

# Fix the numpy random seed for reproducible results
seed = 18945
np.random.seed(seed)

# Some formating options
%config InlineBackend.figure_formats = ['svg']
```

# 1. Building an ensemble of neurons

**a) Tuning curves.** Plot the population tuning curves. Plot the representation accuracy plot ($x - \hat{x}$). Compute and report the RMSE.


```python
def rmse(x1, x2):
    return np.sqrt(np.mean(np.power(x1 - x2, 2)))


# number of neurons
n = 100
tau_rc = 20 / 1000
tau_ref = 2 / 1000
dimensions = 1
encoders = [-1, 1]
model = nengo.Network(label="1-Dim Ensemble", seed=seed)
lif = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)
with model:
    ens = nengo.Ensemble(
        n_neurons=n,
        dimensions=dimensions,
        max_rates=nengo.dists.Uniform(100, 200),
        neuron_type=lif,
    )
    connection = nengo.Connection(ens, ens)

simulation = nengo.Simulator(model)
x, A = tuning_curves(ens, simulation)

plt.figure()
plt.suptitle("1D LIF Neurons")
plt.plot(x, A)
plt.ylabel("Firing Rate (Hz)")
plt.xlabel("Input $x$")
plt.show()


eval_points, targets, decoded = eval_point_decoding(connection, simulation)

plt.figure()
plt.suptitle("$x$ and $\hat{x}$")
plt.plot(targets, decoded)
plt.show()


error = rmse(targets, decoded)
print("RMSE-WITH-100-NEURONS-----------------------")
print(error)
print("--------------------------------------------")
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("bc1ec687-3c77-4e87-b12e-dd6d5f091962");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="bc1ec687-3c77-4e87-b12e-dd6d5f091962" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('bc1ec687-3c77-4e87-b12e-dd6d5f091962');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_4_2.svg)
    



    
![svg](assignment-4_files/assignment-4_4_3.svg)
    


    RMSE-WITH-100-NEURONS-----------------------
    0.003293590017523843
    --------------------------------------------


**b) RMSE and radius.** Compute the RMSE for (at least) the four different radii $0.5$, $1$, $2$, and $4$. Plot your results. Compute the RMSE for (at least) the four different radii $0.5$, $1$, $2$, and $4$. Plot your results. Make sure your neurons have the same (relative, i.e., scaled by the radius) $x$-intercepts and maximum rates across all experiments.


```python
radii = [1 / 2, 1, 2, 4]
errors = []
e_rs = []
for radius in radii:
    ens.radius = radius
    simulation = nengo.Simulator(model)
    x, A = tuning_curves(ens, simulation)
    plt.figure()
    plt.suptitle("100 LIF neurons with radius: " + str(radius))
    plt.plot(x, A)
    plt.xlabel("Input $x$")
    plt.ylabel("Firing Rate (Hz)")
    plt.xlim([-radius, radius])
    plt.show()
    eval_points, targets, decoded = eval_point_decoding(connection, simulation)
    plt.figure()
    plt.suptitle("$x$ and $\hat{x}$ for radius: " + str(radius))
    plt.plot(targets, decoded)
    plt.show()
    rmse_err = rmse(targets, decoded)

    ob = {"rmse": rmse_err, "radius": radius}
    errors.append(ob)
    e_rs.append(rmse_err)

for error in errors:
    print("RMSE-WITH-100-NEURONS---------RADIUS-" + str(error["radius"]) + "------")
    print(error["rmse"])
    print("--------------------------------------------")

plt.figure()
plt.suptitle("Error vs radius")
plt.plot(radii, e_rs)
plt.xlabel("radius")
plt.ylabel("$RMSE$")
plt.show()
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("ae226f71-13d1-45e9-b6b9-97c7d4ac950a");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="ae226f71-13d1-45e9-b6b9-97c7d4ac950a" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('ae226f71-13d1-45e9-b6b9-97c7d4ac950a');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_6_2.svg)
    



    
![svg](assignment-4_files/assignment-4_6_3.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("9ecf134b-076b-40f0-a00c-1aff2805ef1d");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="9ecf134b-076b-40f0-a00c-1aff2805ef1d" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('9ecf134b-076b-40f0-a00c-1aff2805ef1d');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_6_6.svg)
    



    
![svg](assignment-4_files/assignment-4_6_7.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("7f54399b-5b7b-41fa-96c8-ebc412e15e23");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="7f54399b-5b7b-41fa-96c8-ebc412e15e23" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('7f54399b-5b7b-41fa-96c8-ebc412e15e23');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_6_10.svg)
    



    
![svg](assignment-4_files/assignment-4_6_11.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("ca441fcd-c3d4-40ce-bea6-eba77a5e567b");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="ca441fcd-c3d4-40ce-bea6-eba77a5e567b" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('ca441fcd-c3d4-40ce-bea6-eba77a5e567b');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_6_14.svg)
    



    
![svg](assignment-4_files/assignment-4_6_15.svg)
    


    RMSE-WITH-100-NEURONS---------RADIUS-0.5------
    0.0016467950087619215
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------RADIUS-1------
    0.003293590017523843
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------RADIUS-2------
    0.006587180035047686
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------RADIUS-4------
    0.013174360070095372
    --------------------------------------------



    
![svg](assignment-4_files/assignment-4_6_17.svg)
    


**c) Discussion.** What mathematical relationship between the radius and the RMSE do you observe (write down an equation)? Explain why this is the case.

As we can see in the above figure, the relationship between the radius and the RMSE is linear in nature. As a result we can take any two points and use them to create a linear function that would approximate the expected RMSE for a given radius $r$. for the following, let the RMSE $E$ be a linear function $E(r)$ where $m$ and $b$ are costant coefficients. Then:
$$
E = mr+b
$$

$$
 (0.5,0.001646) \to (r_1,E_1) \text{  and  } (4,0.329359) \to (r_2,E_2)
$$

$$
m=\frac{E_2-E_1}{r_2-r_1}=0.09363
$$

$$
b=\frac{r_2E_1-r_1E_2}{r_2-r_1}=-0.04517
$$

So,
$$
E_{RMSE}\approx 0.09363r-0.04517 : r \gt 0
$$

**d) RMSE and refractory period.** What happens to the RMSE and the tuning curves as $\tau_\mathrm{ref}$ changes between $1$ and $5\,\mathrm{ms}$? Plot the tuning curves for at least four different $\tau_\mathrm{ref}$ and produce a plot showing the RMSE over $\tau_\mathrm{ref}$. Again, make sure to use the same neuron ensemble parameters in all your trials.


```python
tau_refs = [1 / 1000, 2 / 1000, 3 / 1000, 4 / 1000, 5 / 1000]
n = 100
tau_rc = 20 / 1000
dimensions = 1
encoders = [-1, 1]

errors = []
e_rs = []

for tau_ref in tau_refs:
    model = nengo.Network(label="1-Dim Ensemble", seed=seed)
    lif = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)
    with model:
        ens = nengo.Ensemble(
            n_neurons=n,
            dimensions=dimensions,
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=lif,
        )
        connection = nengo.Connection(ens, ens)

    simulation = nengo.Simulator(model)
    x, A = tuning_curves(ens, simulation)
    plt.figure()
    plt.suptitle("100 LIF neurons with $\\tau_{ref}$: " + str(tau_ref))
    plt.plot(x, A)
    plt.xlabel("Input $x$")
    plt.ylabel("Firing Rate (Hz)")
    plt.xlim([-1, 1])
    plt.show()
    eval_points, targets, decoded = eval_point_decoding(connection, simulation)
    plt.figure()
    plt.suptitle("$x$ and $\hat{x}$ for $\\tau_{ref}$: " + str(tau_ref))
    plt.plot(targets, decoded)
    plt.show()
    rmse_err = rmse(targets, decoded)

    ob = {"rmse": rmse_err, "tau_ref": tau_ref}
    errors.append(ob)
    e_rs.append(rmse_err)

for error in errors:
    print("RMSE-WITH-100-NEURONS---------TAU_REF-" + str(error["tau_ref"]) + "------")
    print(error["rmse"])
    print("--------------------------------------------")

plt.figure()
plt.suptitle("Error vs $\\tau_{ref}$")
plt.plot(tau_refs, e_rs)
plt.xlabel("$\\tau_{ref}$")
plt.ylabel("$RMSE$")
plt.show()
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("7b20e6d2-c3c1-494a-9b43-f790a3af3e60");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="7b20e6d2-c3c1-494a-9b43-f790a3af3e60" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('7b20e6d2-c3c1-494a-9b43-f790a3af3e60');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_10_2.svg)
    



    
![svg](assignment-4_files/assignment-4_10_3.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("7da6e210-71c1-4a32-a6e5-175a163ba427");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="7da6e210-71c1-4a32-a6e5-175a163ba427" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('7da6e210-71c1-4a32-a6e5-175a163ba427');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_10_6.svg)
    



    
![svg](assignment-4_files/assignment-4_10_7.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("e8fe8472-afb6-4b29-b362-b57fd1fab0e4");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="e8fe8472-afb6-4b29-b362-b57fd1fab0e4" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('e8fe8472-afb6-4b29-b362-b57fd1fab0e4');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_10_10.svg)
    



    
![svg](assignment-4_files/assignment-4_10_11.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("c22497ff-5ec4-4c13-9ac3-07958c8c4c25");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="c22497ff-5ec4-4c13-9ac3-07958c8c4c25" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('c22497ff-5ec4-4c13-9ac3-07958c8c4c25');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_10_14.svg)
    



    
![svg](assignment-4_files/assignment-4_10_15.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("dfe244ed-2d4f-43f9-9e26-0cd736b4171f");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="dfe244ed-2d4f-43f9-9e26-0cd736b4171f" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('dfe244ed-2d4f-43f9-9e26-0cd736b4171f');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_10_18.svg)
    



    
![svg](assignment-4_files/assignment-4_10_19.svg)
    


    RMSE-WITH-100-NEURONS---------TAU_REF-0.001------
    0.0029890403235803675
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------TAU_REF-0.002------
    0.003293590017523843
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------TAU_REF-0.003------
    0.003947854781132681
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------TAU_REF-0.004------
    0.005210890671128928
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------TAU_REF-0.005------
    0.008031351083449448
    --------------------------------------------



    
![svg](assignment-4_files/assignment-4_10_21.svg)
    


**e) RMSE and membrane time constant.** What happens to the RMSE and the tuning curves as $\tau_\mathrm{RC}$ changes between $10$ and $100\,\mathrm{ms}$? Plot the tuning curves for at least four different $\tau_\mathrm{RC}$ and produce a plot showing the RMSE over $\tau_\mathrm{RC}$.  Again, make sure to use the same neuron ensemble parameters in all your trials.


```python
tau_rcs = [10 / 1000, 25 / 1000, 50 / 1000, 75 / 1000, 100 / 1000]
n = 100
tau_ref = 2 / 1000
dimensions = 1
encoders = [-1, 1]

errors = []
e_rs = []

for tau_rc in tau_rcs:
    model = nengo.Network(label="1-Dim Ensemble", seed=seed)
    lif = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)
    with model:
        ens = nengo.Ensemble(
            n_neurons=n,
            dimensions=dimensions,
            max_rates=nengo.dists.Uniform(100, 200),
            neuron_type=lif,
        )
        connection = nengo.Connection(ens, ens)

    simulation = nengo.Simulator(model)
    x, A = tuning_curves(ens, simulation)
    plt.figure()
    plt.suptitle("100 LIF neurons with $\\tau_{rc}$: " + str(tau_rc))
    plt.plot(x, A)
    plt.xlabel("Input $x$")
    plt.ylabel("Firing Rate (Hz)")
    plt.xlim([-1, 1])
    plt.show()
    eval_points, targets, decoded = eval_point_decoding(connection, simulation)
    plt.figure()
    plt.suptitle("$x$ and $\hat{x}$ for $\\tau_{rc}$: " + str(tau_rc))
    plt.plot(targets, decoded)
    plt.show()
    rmse_err = rmse(targets, decoded)

    ob = {"rmse": rmse_err, "tau_rc": tau_rc}
    errors.append(ob)
    e_rs.append(rmse_err)

for error in errors:
    print("RMSE-WITH-100-NEURONS---------TAU_RC-" + str(error["tau_rc"]) + "------")
    print(error["rmse"])
    print("--------------------------------------------")

plt.figure()
plt.suptitle("Error vs $\\tau_{rc}$")
plt.plot(tau_refs, e_rs)
plt.xlabel("$\\tau_{rc}$")
plt.ylabel("$RMSE$")
plt.show()
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("a158ac84-5929-41e6-8618-2176a933a82c");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="a158ac84-5929-41e6-8618-2176a933a82c" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('a158ac84-5929-41e6-8618-2176a933a82c');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_12_2.svg)
    



    
![svg](assignment-4_files/assignment-4_12_3.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("459b6186-f7c0-4b74-8c8b-9c0116a79768");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="459b6186-f7c0-4b74-8c8b-9c0116a79768" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('459b6186-f7c0-4b74-8c8b-9c0116a79768');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_12_6.svg)
    



    
![svg](assignment-4_files/assignment-4_12_7.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("b89b427f-8562-40f4-9927-ddd2096bece1");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="b89b427f-8562-40f4-9927-ddd2096bece1" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('b89b427f-8562-40f4-9927-ddd2096bece1');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_12_10.svg)
    



    
![svg](assignment-4_files/assignment-4_12_11.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("e0af7cf7-0ae1-4710-9edb-102d5e549cf6");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="e0af7cf7-0ae1-4710-9edb-102d5e549cf6" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('e0af7cf7-0ae1-4710-9edb-102d5e549cf6');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_12_14.svg)
    



    
![svg](assignment-4_files/assignment-4_12_15.svg)
    




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("c346f61f-953f-43f9-94de-963aadb726a7");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="c346f61f-953f-43f9-94de-963aadb726a7" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('c346f61f-953f-43f9-94de-963aadb726a7');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_12_18.svg)
    



    
![svg](assignment-4_files/assignment-4_12_19.svg)
    


    RMSE-WITH-100-NEURONS---------TAU_RC-0.01------
    0.0043980089499409455
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------TAU_RC-0.025------
    0.0030491458286627175
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------TAU_RC-0.05------
    0.002582404009989868
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------TAU_RC-0.075------
    0.0024547862794107003
    --------------------------------------------
    RMSE-WITH-100-NEURONS---------TAU_RC-0.1------
    0.002402380162489991
    --------------------------------------------



    
![svg](assignment-4_files/assignment-4_12_21.svg)
    


**f) Discussion.** Discuss the last two results. Describe what happens to the tuning curves as $\tau_\mathrm{ref}$ and $\tau_\mathrm{RC}$ change (you do not need to come up with a mathematical relationship here). Explain why the change in tuning curve shape influences the RMSE in the way you observe.

As $\tau_{ref}$ increases we can see that that the $RMSE$ increases in what appears to be an exponential nature. This is likely due the fact that the refractory period is limiting the rate at which the neurons can "spike" so as we increase the period, we would expect to see an increase in information loss in the signal because the neurons can't fire at a rate at that preserves much of the input signal.

In the case of the membrane time constant $\tau_{rc}$ we can see that the error decreases exponentially and decays to $0$ as $\tau_{rc}$ increases. This is because at low $\tau_{rc}$ values, neurons get excited very quickly and this can cause them to fire at incorrect times increasing noise in the neuron's output. As $\tau_{rc}$ increases, neurons become less easily excited and "noisey" the neurons become. At a reasonable $\tau_{ref}$ this results in a less noisey output signal.



# 2. Connecting neurons

**a) Computing the identity function.** Show the input value and the decoded values from the two  ensembles in three separate plots. Run the simulation for $0.5\,\mathrm{s}$.


```python
# number of neurons
n_1 = 100
n_2 = 50
tau_rc = 20 / 1000
tau_ref = 2 / 1000
dimensions = 1
encoders = [-1, 1]
model = nengo.Network(label="1-Dim Ensemble", seed=seed)
lif = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)


step_fn = lambda t: 0 if t < 0.1 else (1.0 if t < 0.4 else 0)
with model:
    x = nengo.Node(step_fn)
    A = nengo.Ensemble(
        n_neurons=n_1,
        dimensions=dimensions,
        max_rates=nengo.dists.Uniform(100, 200),
    )
    B = nengo.Ensemble(
        n_neurons=n_2,
        dimensions=dimensions,
        max_rates=nengo.dists.Uniform(100, 200),
    )
    con_stim_ens_1 = nengo.Connection(x, A)
    con_ens_1_ens_2 = nengo.Connection(A, B, synapse=10 / 1000)
    # create probes so we know what is going on
    probe_x = nengo.Probe(x)
    probe_A = nengo.Probe(A, synapse=10 / 1000)
    probe_B = nengo.Probe(B, synapse=10 / 1000)

simulation = nengo.Simulator(model)

run_time = 0.5
simulation.run(run_time)

t = simulation.trange()
# input
plt.figure()
plt.suptitle("Step input $x(t)$ into Population $A$")
plt.plot(t, simulation.data[probe_x])
plt.xlabel("$t$")
plt.ylabel("$x(t)")
plt.show()

# population A
plt.figure()
plt.suptitle("Decoded Output from Population $A$")
plt.plot(t, simulation.data[probe_A])
plt.ylabel("$\hat{x}_{A}$")
plt.xlabel("$t$")
plt.show()

# population B
plt.figure()
plt.suptitle("Decoded Output from Population $B$")
plt.plot(t, simulation.data[probe_B])
plt.ylabel("$\hat{y}_{B}$")
plt.xlabel("$t$")
plt.show()
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("fc3265b4-846e-4fd1-a55b-0278cdc54a39");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="fc3265b4-846e-4fd1-a55b-0278cdc54a39" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('fc3265b4-846e-4fd1-a55b-0278cdc54a39');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("f67416c6-882e-46cc-8d51-482b8d19d155");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="f67416c6-882e-46cc-8d51-482b8d19d155" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('f67416c6-882e-46cc-8d51-482b8d19d155');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Simulation finished in 0:00:01.';

            if (100.0 > 0.) {
                fill.style.transition = 'width 0.1s linear';
            } else {
                fill.style.transition = 'none';
            }

            fill.style.width = '100.0%';
            fill.style.animation = 'none';
            fill.style.backgroundImage = 'none'


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_16_4.svg)
    



    
![svg](assignment-4_files/assignment-4_16_5.svg)
    



    
![svg](assignment-4_files/assignment-4_16_6.svg)
    


**b) Computing an affine transformation.** Make a new version of the model where instead of computing the identity function, it computes $y(t) = 1 - 2x(t)$. Show the same graphs as in part (a).


```python
# number of neurons
n_1 = 100
n_2 = 50
tau_rc = 20 / 1000
tau_ref = 2 / 1000
dimensions = 1
encoders = [-1, 1]
model = nengo.Network(label="1-Dim Ensemble", seed=seed)
lif = nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref)


step_fn = lambda t: 0 if t < 0.1 else (1.0 if t < 0.4 else 0)
func = lambda x: 1 - 2 * x
with model:
    x = nengo.Node(step_fn)
    A = nengo.Ensemble(
        n_neurons=n_1,
        dimensions=dimensions,
        max_rates=nengo.dists.Uniform(100, 200),
    )
    B = nengo.Ensemble(
        n_neurons=n_2,
        dimensions=dimensions,
        max_rates=nengo.dists.Uniform(100, 200),
    )
    con_stim_ens_1 = nengo.Connection(x, A)
    con_ens_1_ens_2 = nengo.Connection(A, B, synapse=10 / 1000, function=func)
    # create probes so we know what is going on
    probe_x = nengo.Probe(x)
    probe_A = nengo.Probe(A, synapse=10 / 1000)
    probe_B = nengo.Probe(B, synapse=10 / 1000)

simulation = nengo.Simulator(model)

run_time = 0.5
simulation.run(run_time)

t = simulation.trange()
# input
plt.figure()
plt.suptitle("Step input $x(t)$ into Population $A$")
plt.plot(t, simulation.data[probe_x])
plt.xlabel("$t$")
plt.ylabel("$x(t)")
plt.show()

# population A
plt.figure()
plt.suptitle("Decoded Output from Population $A$")
plt.plot(t, simulation.data[probe_A])
plt.ylabel("$\hat{x}_{A}$")
plt.xlabel("$t$")
plt.show()

# population B
plt.figure()
plt.suptitle("Decoded Output from Population $B$ computing $y(t)=1-2x(t)$")
plt.plot(t, simulation.data[probe_B])
plt.ylabel("$\hat{y}_{B}$")
plt.xlabel("$t$")
plt.show()
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("b1f992fd-d2e8-471b-8ce1-8d2939793c3c");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="b1f992fd-d2e8-471b-8ce1-8d2939793c3c" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('b1f992fd-d2e8-471b-8ce1-8d2939793c3c');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("b6e35219-fb01-4856-aef2-3370975089f8");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="b6e35219-fb01-4856-aef2-3370975089f8" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('b6e35219-fb01-4856-aef2-3370975089f8');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Simulation finished in 0:00:01.';

            if (100.0 > 0.) {
                fill.style.transition = 'width 0.1s linear';
            } else {
                fill.style.transition = 'none';
            }

            fill.style.width = '100.0%';
            fill.style.animation = 'none';
            fill.style.backgroundImage = 'none'


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_18_4.svg)
    



    
![svg](assignment-4_files/assignment-4_18_5.svg)
    



    
![svg](assignment-4_files/assignment-4_18_6.svg)
    


# 3. Dynamics

**a) Transforming the dynamical system.** Rewrite the linear dynamical system describing the integrator in terms of $\frac{\mathrm{d}\vec x(t)}{\mathrm{d}t} = \mathbf{A} \mathbf{x} + \mathbf{B} \mathbf{u}$, i.e., write down the matrices $\mathbf{A}$ and $\mathbf{B}$ (you can just use the equations from class, you do not have to re-derive the equations) What are the matrices $\mathbf{A}'$ and $\mathbf{B}'$ we have to use when implementing this system using the recurrent connection post-synaptic filter?

Beginning with the linear, time invariant system $\phi(\vec u,\vec x)=\bold A'\vec x + \bold B'\vec u$, we can write this system in the _time-domain_ as:

$$
\frac{d}{dt}\vec x(t) = \bold A \vec x(t) + \bold B \vec u(t)
$$
and
$$
\vec x(t) = (h \ast (\bold A'\vec x+\bold B \vec u))(t)
$$
We can find $\bold A'$ and $\bold B'$ by converting to the _Laplace-domain_ as follows:

$$
sX(s)=\bold A X(s) + \bold B U(s)
$$
and
$$
X(s)=H(s)(\bold A' X(s)+ \bold B' U(s))
$$

given that

$$
H(s) = \frac{1}{1+s \tau}
$$

we can solve the above equations for the matrices $\bold A'$ and $\bold B'$ given that we now have three equations, and three unknowns. This results in the following:

$$
\bold A' = \tau \bold A + \bold I
$$
and 

$$
\bold B' = \tau \bold B
$$

**In the case of the linear integrator, we see that**

$$
\frac{dx(t)}{dt} = \vec u
$$
Writing the integrator in the above canonical form, the above equation is true when $\bold A = 0$ and $\bold B = \bold I$, therefore we using the previous equations for $\bold A'$ and $\bold B'$ we can see that:

$$
\bold A' = \bold I
$$
and
$$
\bold B' = \tau \bold I
$$

**b) Integrator using spiking neurons.**  Show the input, the ideal integral, and the value represented by the ensemble when the input is a value of $0.9$ from $t=0.04$ to $t=1.0$ (and $0$ for other times). Run the simulation for $1.5\,\mathrm{s}$.


```python


def simulate(
    input=lambda t: 0.9 if t >= 0.04 and t <= 1.0 else 0,
    run_time=1.5,
    tau=50 / 1000,
    title="Neural Integrator for a step input",
):
    n_neurons = 200
    model = nengo.Network(label="Integrator")
    dimensions = 1
    transform = 1 * tau

    def recur(f):
        return 1 * f

    with model:
        x = nengo.Node(input)
        ensemble = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=dimensions,
            max_rates=nengo.dists.Uniform(100, 200),
        )
        nengo.Connection(x, ensemble, transform=transform, synapse=5 / 1000)
        nengo.Connection(ensemble, ensemble, function=recur, synapse=50 / 1000)
        probe_x = nengo.Probe(x, synapse=10 / 1000)
        probe_ensemble = nengo.Probe(ensemble, synapse=10 / 1000)

    simulation = nengo.Simulator(model)
    # run for 1.5s

    simulation.run(run_time)

    t = simulation.trange()

    time = np.arange(0, run_time, run_time / 1500)

    f = [input(t) for t in time]

    ideal = np.cumsum(f) * (run_time / 1500)

    plt.figure()
    plt.suptitle(title)
    a = plt.plot(t, simulation.data[probe_x], label="$x(t)$")
    b = plt.plot(
        t, simulation.data[probe_ensemble], label="$\hat{\int_{0}^{1.5}{x(t)}}$"
    )
    c = plt.plot(t, ideal, label="Ideal $\int_{0}^{1.5}{x(t)}}$")
    plt.legend(
        handles=[a, b, c],
        labels=[],
    )
    plt.xlim([0, 1.5])
    plt.show()


simulate()
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("eea9471b-0ed4-43b1-9874-1a2eb433e1cc");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="eea9471b-0ed4-43b1-9874-1a2eb433e1cc" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('eea9471b-0ed4-43b1-9874-1a2eb433e1cc');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("676ea600-ee62-4744-9c25-1ac5b61a8564");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="676ea600-ee62-4744-9c25-1ac5b61a8564" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('676ea600-ee62-4744-9c25-1ac5b61a8564');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Simulation finished in 0:00:01.';

            if (100.0 > 0.) {
                fill.style.transition = 'width 0.1s linear';
            } else {
                fill.style.transition = 'none';
            }

            fill.style.width = '100.0%';
            fill.style.animation = 'none';
            fill.style.backgroundImage = 'none'


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_22_4.svg)
    


**c) Discussion.** What is the expected ideal result, i.e., if we just mathematically computed the integral of the input, what is the equation describing the integral? How does the simulated output compare to that ideal?


The computed result looks quite similar to what we would expect to see if we just computed the integral. We would expect to see a linear increasing function over the region for which the input is non-zero and as time went to infinity, we would expect to see that value to remain constant. i.e


...if $t \lt 0.04$
$$
\int_{0}^{t} u(t')dt' = 0
$$
...if $ 0.04 \leq t \lt 1$
$$
\int_{0}^{t} u(t')dt' = 0.9t
$$
... if $ 1 \leq t$
$$
\int_{0}^{t} u(t')dt' = [0.9t]_{0.04}^{1} = 0.864
$$

**d) Simulation using rate neurons.** Change the neural simulation to rate mode. Re-run the simulation in rate mode. Show the resulting plots.


```python
n_neurons = 200
model = nengo.Network(label="Integrator")
tau_rc = 20 / 1000
tau_ref = 2 / 1000
dimensions = 1
tau = 50 / 1000
synapse = 5 / 1000
input = lambda t: 0.9 if t >= 0.04 and t <= 1.0 else 0
transform = 1 * tau


def recur(f):
    return 1 * f

with model:
    x = nengo.Node(input)
    ensemble = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=dimensions,
        max_rates=nengo.dists.Uniform(100, 200),
        neuron_type=nengo.LIFRate(),
    )
    con_in = nengo.Connection(x, ensemble, transform=transform, synapse=5 / 1000)
    con_recur = nengo.Connection(ensemble, ensemble, function=recur, synapse=50 / 1000)
    probe_x = nengo.Probe(x, synapse=10 / 1000)
    probe_ensemble = nengo.Probe(ensemble, synapse=10 / 1000)

simulation = nengo.Simulator(model)
# run for 1.5s
run_time = 1.5
simulation.run(run_time)

t = simulation.trange()

plt.figure()
plt.suptitle("Neural Integrator for a step input")
a = plt.plot(t, simulation.data[probe_x], label="$x(t)$")
b = plt.plot(t, simulation.data[probe_ensemble], label="$\hat{\int_{0}^{1.5}{x(t)}}$")
plt.legend(
    handles=[
        a,
        b,
    ],
    labels=[],
)
plt.xlim([0, 1.5])
plt.show()
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("a55eb186-db5e-41e7-85b6-3bc03627f700");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="a55eb186-db5e-41e7-85b6-3bc03627f700" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('a55eb186-db5e-41e7-85b6-3bc03627f700');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("a5012b0a-f1fb-41b7-be7a-3a05c452b872");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="a5012b0a-f1fb-41b7-be7a-3a05c452b872" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('a5012b0a-f1fb-41b7-be7a-3a05c452b872');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Simulation finished in 0:00:01.';

            if (100.0 > 0.) {
                fill.style.transition = 'width 0.1s linear';
            } else {
                fill.style.transition = 'none';
            }

            fill.style.width = '100.0%';
            fill.style.animation = 'none';
            fill.style.backgroundImage = 'none'


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_26_4.svg)
    


**e) Discussion.** How does this compare to the result in part (b)? What deviations from the ideal do you still observe? Where do those deviations come from?

While adding the `LIFRate()` seems to run the simulation in `rate mode` seems to have signifnicantly decreased the amount of noise in the simulation, it also appears to be an underestimate of the actual integral. This is likely because there is inevitably some information that is lost when the signal undergoes filtering.  

**f) Integration of a shorter input pulse.** Returning to spiking mode, change the input to be a value of $0.9$ from $t=0.04$ to $0.16$. Show the same plots as before (the input, the ideal, and the value represented by the ensemble over $1.5\,\mathrm{s}$).


```python
input = lambda t: 0.9 if t >= 0.04 and t <= 0.16 else 0
simulate(input=input)
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("0730f383-51de-4719-9ffe-b730c3e37d26");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="0730f383-51de-4719-9ffe-b730c3e37d26" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('0730f383-51de-4719-9ffe-b730c3e37d26');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("db9cd09a-72ab-4962-84fd-528e338afbfc");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="db9cd09a-72ab-4962-84fd-528e338afbfc" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('db9cd09a-72ab-4962-84fd-528e338afbfc');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Simulation finished in 0:00:01.';

            if (100.0 > 0.) {
                fill.style.transition = 'width 0.1s linear';
            } else {
                fill.style.transition = 'none';
            }

            fill.style.width = '100.0%';
            fill.style.animation = 'none';
            fill.style.backgroundImage = 'none'


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_30_4.svg)
    


**g) Discussion.** How does this compare to (b)? What is the ideal equation? Does it work as intended? If not, why is it better or worse?

In the above case, the integrator performs significantly worse. This is likely due to the fact that the integrator is bound by it's nuuron's max firing rates and refractory periods. As a result, the integrator can't "fire fast enough" to represent the signal before the step function is "turned off".

The ideal equation is as follows

...if $t \lt 0.04$
$$
\int_{0}^{t} u(t')dt' = 0
$$
...if $ 0.04 \leq t \lt 0.16$
$$
\int_{0}^{t} u(t')dt' = 0.9t
$$
... if $ 0.16 \leq t$
$$
\int_{0}^{t} u(t')dt' = [0.9t]_{0.04}^{0.16} = 0.108
$$

**h) Input ramp.** Change the input to a ramp input from $0$ to $0.9$ from $t=0$ to $t=0.45$ (and $0$ for $t>0.45$). Show the same plots as in the previous parts of this question.


```python
input = lambda t: 2*t if t >= 0 and t <= 0.45 else 0
simulate(input=input,title="Neural Integrator for a Ramp Input")
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("4b003362-059e-4082-a02e-24ab0d28f3a7");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="4b003362-059e-4082-a02e-24ab0d28f3a7" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('4b003362-059e-4082-a02e-24ab0d28f3a7');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("c65f4adf-b533-4b1c-896a-b2339d5624bd");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="c65f4adf-b533-4b1c-896a-b2339d5624bd" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('c65f4adf-b533-4b1c-896a-b2339d5624bd');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Simulation finished in 0:00:01.';

            if (100.0 > 0.) {
                fill.style.transition = 'width 0.1s linear';
            } else {
                fill.style.transition = 'none';
            }

            fill.style.width = '100.0%';
            fill.style.animation = 'none';
            fill.style.backgroundImage = 'none'


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_34_4.svg)
    


**i) Discussion.** What does the ensemble end up representing, and why? What is the (ideal) equation for the curve traced out by the ensemble?

The output looks like a qudratic function over the regions for which the ramp input $x(t)=2t$ is increasing, and then levels off when $x(t)=0$ after 0.45 seconds. This is exactly what we expect to see since the antiderivative of a linear function is quadratic, indicating that the model is in fact calculating an integral. It does once again reveal the limitations of the firing rate of the neurons in representing inregrals on small timescales. We again see that:

The ideal equation is as follows

... if  $0 \lt t \leq 0.45$
$$
\int_{0}^{t} u(t')dt' = t^2
$$
... if $ 0.45 \lt t$
$$
\int_{0}^{t} u(t')dt' = [t^2]_{0}^{0.45} = 0.2025
$$

**j) Sinusoidal input.** Change the input to $5\sin(5t)$. Show the same plots as before.


```python
input = lambda t: 5 * np.sin(5 * t)
simulate(input=input, title="Neural Integrator for a sinusoidal Input")
```



<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("605051db-df08-48d4-8e33-bee17d55ab73");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="605051db-df08-48d4-8e33-bee17d55ab73" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('605051db-df08-48d4-8e33-bee17d55ab73');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Build finished in 0:00:01.';

            fill.style.width = '100%';
            fill.style.animation = 'pb-fill-anim 2s linear infinite';
            fill.style.backgroundSize = '100px 100%';
            fill.style.backgroundImage = 'repeating-linear-gradient(' +
                '90deg, #bdd2e6, #edf2f8 40%, #bdd2e6 80%, #bdd2e6)';


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>




<script>
    if (Jupyter.version.split(".")[0] < 5) {
        var pb = document.getElementById("f62667d2-9006-4d29-8f4b-a0835331f443");
        var text = document.createTextNode(
            "HMTL progress bar requires Jupyter Notebook >= " +
            "5.0 or Jupyter Lab. Alternatively, you can use " +
            "TerminalProgressBar().");
        pb.parentNode.insertBefore(text, pb);
    }
</script>
<div id="f62667d2-9006-4d29-8f4b-a0835331f443" style="
    width: 100%;
    border: 1px solid #cfcfcf;
    border-radius: 4px;
    text-align: center;
    position: relative;">
  <div class="pb-text" style="
      position: absolute;
      width: 100%;">
    0%
  </div>
  <div class="pb-fill" style="
      background-color: #bdd2e6;
      width: 0%;">
    <style type="text/css" scoped="scoped">
        @keyframes pb-fill-anim {
            0% { background-position: 0 0; }
            100% { background-position: 100px 0; }
        }
    </style>
    &nbsp;
  </div>
</div>



<script>
              (function () {
                  var root = document.getElementById('f62667d2-9006-4d29-8f4b-a0835331f443');
                  var text = root.getElementsByClassName('pb-text')[0];
                  var fill = root.getElementsByClassName('pb-fill')[0];

                  text.innerHTML = 'Simulation finished in 0:00:01.';

            if (100.0 > 0.) {
                fill.style.transition = 'width 0.1s linear';
            } else {
                fill.style.transition = 'none';
            }

            fill.style.width = '100.0%';
            fill.style.animation = 'none';
            fill.style.backgroundImage = 'none'


                fill.style.animation = 'none';
                fill.style.backgroundImage = 'none';

              })();
        </script>



    
![svg](assignment-4_files/assignment-4_38_4.svg)
    


**k) Discussion.** What should the value represented by the ensemble be? Write the equation. How well does it do? What are the differences between the model's behaviour and the expected ideal behaviour and why do these differences occur?

The model does a relatively good job of representing the sinusodial input. It shows an estimation of correct amplitude and period. With that beign said, the integrator only "adds" at the beginning since it has no prior information to go off of, and as a result it becomes out of phase with the actual function $x(t)=5sin(5t)$

The ideal results are as follows

... if $ 0 \lt t$
$$
\int_{0}^{t} u(t')dt' = \int_{0}^{t'} 5sin(5t)dt = -cos(5t)+1
$$
Or on the interval $[0,1.5]$:
$$
\int_{0}^{t} u(t')dt' = \int_{0}^{1.5} 5sin(5t)dt = 5 \int_{0}^{7.5}sin(u)\frac{1}{5}du = [-cos(u)]_{0}^{7.5} = 0.65336...
$$

**l) 🌟 Bonus question.** Implement a nonlinear dynamical system we have not seen in class (and that is not in the book). Demonstrate that it's working as expected

😥
