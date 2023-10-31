Direct integration method for Rayleigh/Scholte wave dispersion and eigenfunctions in a layered half-space with or without a water layer. This largely follows Takeuchi & Saito (1972) or equivalently Section 7.2.1 of Aki & Richards (2002). Two independent solutions are integrated from the bottom layer to the surface of the elastic half-space, a propagator matrix is appended to account for the water layer, and then the linear combination is found which satisfies the stress boundary conditions.

This is (at least) 10x slower than CPS and much less numerically stable at low frequencies. However, it seems to be more numerically stable at high frequencies and for smooth (e.g. gradient or power-law) velocity models.

Ethan Williams
2023-10-31  
