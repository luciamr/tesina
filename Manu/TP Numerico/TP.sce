//Parte B
//Ejercicio 1

m = 1.83;
k = 220;
c = 16.05;
dMax = 3/100;
v = 70*1000/3600;
cantTramos = 8;
longTramos = 12;

function r = d2Ug(t, v)
    extra = %pi/6 * v;
    r = dMax/2 * extra^2 * cos(t * extra);
endfunction;

function dX = f(t, X, v)
    dX(1) = X(2);
    if v*t <= cantTramos * longTramos then
        dX(2) = -(m * d2Ug(t, v) + c * X(2) + k * X(1))/m;
    else dX(2) = -(c * X(2) + k * X(1))/m;
    end;
endfunction;

X0 = [0; 0];
ti = 0;
tf = 20;
paso = .001;
t = ti:paso:tf;

Y = ode("rk", X0, ti, t, list(f, v));

//Parte 1.a
scf(0);
plot(t, Y(1, :), "c"); //desplazamiento vertical
xlabel("Tiempo (s)", 'fontsize', 2);
ylabel("Desplazamiento (m)", 'fontsize', 2);
title("Historia del Desplazamiento", 'fontsize', 4);

//Parte 1.b
maxDesp = max(abs(Y(1, :)));
disp(maxDesp);

//Ejercicio 2

vi = 20*1000/3600;
vf = 120*1000/3600;
pasoVelos = 5*1000/3600;
velos = vi:pasoVelos:vf; //velocidades
velMax = [];


for i = 1:length(velos)
    Yi = ode("rk", X0, ti, t, list(f, velos(i)));
    maxDesp = max(abs(Yi(1, :)));
    velMax(i) = maxDesp;
end;

scf(1);
plot(velos*3600/1000, velMax, '--b.');
xlabel("Velocidad (km/h)", 'fontsize', 2);
ylabel("Máximo Desplazamiento (m)", 'fontsize', 2);
title("Valor Máximo del Desplazamiento", 'fontsize', 4);
