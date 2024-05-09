# stellar_evol: The evolution of the stellar mass
Ziyi Guo - [zyguo@smail.nju.edu.cn]; [zyguo.astro@gmail.com]

Hi all, this is a toy model for calculating the stellar mass evolution of a galaxy. 
In this model, we do not consider the influence of metallicity on the stellar lifetime and yield.

After downloading this file, you can use it by the following process:

```python
from stellar_evol import evol as se
se1 = se.stellar_evol(sfh = sfh, lifetime = lifetime, yield_list = yield_list)
```

Input the sfh as a 2\*X array, the first column is the time, and the second column is the star formation rate at this time.
Input the lifetime as a 2\*Y array, the first column is the initial stellar mass, and the second column is the stellar lifetime of this mass.
Input the yield_list as a 2\*Z array, the first column is the initial stellar mass, and the second column is the total yield mass.

All these values can be obtained from the NuPyCEE or other stellar models.

This code will calculate the mass and number of alive stars.

For more information, please read the comments in the evol.py.
