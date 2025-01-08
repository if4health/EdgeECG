%% EXPERIMENTO 9.42
clear 
clc
close all
load('100m');
xn=val(2,:)/200;

%calculando coeficientes para função DB.
a=(1-sqrt(3))/(4*sqrt(2));
b=(3-sqrt(3))/(4*sqrt(2));
c=(3+sqrt(3))/(4*sqrt(2));
d=(1+sqrt(3))/(4*sqrt(2));


%teste dos coeficientes 
p=a.^2+b.^2+c.^2+d.^2; 
%p=1- Dessa forma é possível que y seja apenas um delay de x 

%vetores dos filtros
hplivro=[d -c b -a]; %sem o comparativo
ihplivro=[-a b -c d];%sem o comparativo
hp=[-d c -b a];
ihp=[a -b c -d]; 



lplivro=[a b c d]; %sem o comparativo
ilplivro=[d c b a];%sem o comparativo
lp=[-a -b -c -d]; 
ilp=[-d -c -b -a];

%filtragem
%aplicação direta da função para convoluir:
znc=conv(xn,hp); 
wnc=conv(xn,lp);

%Convolução sem a função
for i=1:3600 
    if i==1 
        wn(1,i) = -a*xn(i);
    else
        if i==2
        wn(1,i) = -a*xn(i) - b*xn(i-1);
        else
            if i==3
        wn(1,i) = -a*xn(i) - b*xn(i-1) - c*xn(i-2);
              else
    wn(1,i) = -a*xn(i) - b*xn(i-1) - c*xn(i-2) - d*xn(i-3); %n eh o do livro
    end
    end
    end
end
for i=1:3600 
    if i==1 
        zn(1,i) = -d*xn(i);
    else
        if i==2
        zn(1,i) = -d*xn(i) + c*xn(i-1);
        else
            if i==3
        zn(1,i) = -d*xn(i) + c*xn(i-1) - b*xn(i-2);
              else
    zn(i) = -d*xn(i) + c*xn(i-1) - b*xn(i-2) + a*xn(i-3); % n eh o do livro
    end
    end
    end
end

figure('name','verificando função com convolução e sem lp')
plot(wn); hold on
plot(wnc); hold off
legend('wn','wnc "Convoluído passa baixa" '); 

figure('name','verificando função com convolução e sem hp')
plot(zn); hold on
plot(znc); hold off
legend('zn','znc "Convoluído passa baixa" '); 

%Resultado ok para convolução sem função!


figure('name','Saídas da 1° filtragem com convolução sem func')

subplot(2,1,1), plot(wn); hold on
legend('wn');
title('Passa baixas')

subplot(2,1,2),plot(zn); 
legend('zn'); hold off
title('Passa altas')

%% componentes da reconstrução do sinal com conv a partir da func sem a
%decimação para provar o dalay
z=conv(zn,ihp);
w=conv(wn,ilp);



%%componentes p/ decimação por partes
%Filtro passa baixa-coeficientes de aproximação 

%apenas a convolução com decomposição por 2:
cont=1;
for i=1:3600
  if mod(i,2)==0  
      
    wd(1,i-cont)=wn(1,i);
    cont= cont+1; 
    
  end
end

figure('name','Compara sinal com e sem decimação')
subplot(3,1,1), plot(wd); hold on
legend('wd "Sinal com decimação"');



%reconstrução da decimação
cont=1;
for i=1:3600
  if mod(i,2)==0  
      
    %wu1(1,i)=wn(1,i); %testar a aplicaçao de wd sem decimar:
     wu(1,i)=wd(1,i-cont);
     cont= cont+1;
else
    wu(1,i)=0;
   %wu1(1,i)=0;
  end
end

subplot(3,1,2), plot(wu); 
legend('wu "Recontrução da decimação"'); 
%plot(wu1); 


%convolução sem func p/ reconstrução:
for i=1:3600 
    if i==1 
        wf(1,i) = -d*wu(i);
    else
        if i==2
        wf(1,i) = -d*wu(i) - c*wu(i-1);
        else
            if i==3
        wf(1,i) = -d*wu(i) - c*wu(i-1) - b*wu(i-2);
              else
    wf(1,i) = -d*wu(i) - c*wu(i-1) - b*wu(i-2) - a*wu(i-3);
    end
    end
    end
end
subplot(3,1,3),plot(wf); 
legend('wf "Componente do Lp do sinal"'); hold off



%Decomposição a partir do high-pass:
cont=1;
for i=1:3600
  if mod(i,2)==0  
      
    zd(1,i-cont)=zn(1,i);
    cont= cont+1;
  else
    
    
  end
end

figure(3)
subplot(3,1,1),plot(zd); hold on
legend('zd "Sinal com decimação"');

cont=1;
for i=1:3600
  if mod(i,2)==0  %dec. com posições pares 
      
%zu(1,i)=zn(1,i);
zu(1,i)=zd(1,i-cont);
     cont= cont+1;
else
    zu(1,i)=0;
    
  end
end

subplot(3,1,2),plot(zu);
legend('zu "Reconstrução da decimação"'); 


%convolução sem função p/ reconstrução:
for i=1:3600 
    if i==1 
        zf(1,i) = -d*zu(i);
    else
        if i==2
        zf(1,i) = -d*zu(i) + c*zu(i-1);
        else
            if i==3
        zf(1,i) = -d*zu(i) + c*zu(i-1) - b*zu(i-2);
              else
    zf(1,i) = a*zu(i) - b*zu(i-1) + c*zu(i-2) - d*zu(i-3);
    end
    end
    end
end

subplot(3,1,3),plot(zf);
legend('zf "Componente do Hp do sinal"'); hold off


% teste da recontrução com a func para 1 decomposição ok

zf2=conv(zu,ihp);
wf2=conv(wu,ilp);
yn2=zf2+wf2; %sinal com a função pronta

% Gráfico 1: Decomposição direta com aplicação da função Convolução
figure('Name', 'Decomposição Direta - Aplicação da Função Convolução', 'NumberTitle', 'off');
plot(xn, 'LineWidth', 0.5); hold on;
plot(yn2, '--', 'LineWidth', 0.5); hold off;
title('Decomposição Direta: Aplicação da Função Convolução');
xlabel('Índice de Amostra');
ylabel('Amplitude');
legend('Sinal Original (xn)', 'Sinal Processado (yn2)', 'Location', 'best');
grid on;

% Gráfico 2: Verificação com e sem convolução
figure('Name', 'Verificação da Função: Com Convolução e Sem Convolução', 'NumberTitle', 'off');
yn3 = zf + wf;
plot(yn3, 'LineWidth', 1.5); hold on;
plot(yn2, '--', 'LineWidth', 1.5); hold off;
title('Comparação: Com Convolução e Sem Convolução');
xlabel('Índice de Amostra');
ylabel('Amplitude');
legend('Sem Convolução (yn3)', 'Com Convolução (yn2)', 'Location', 'best');
grid on;

% Gráfico 3: Decomposição função sem convolução pronta
figure('Name', 'Reconstrução do sinal - 1° decomposição com função manual', 'NumberTitle', 'off');
plot(xn, 'LineWidth', 0.5); hold on;
plot(yn3, '--', 'LineWidth', 0.5); hold off;
title('Reconstrução com 1 decomposição: Aplicação sem a Função Convolução');
xlabel('Índice de Amostra');
ylabel('Amplitude');
legend('Sinal Original (xn)', 'Sinal Processado (yn3)', 'Location', 'best');
grid on;

% Comprimento do sinal original
N = length(yn3);

% Inicializar vetor para o novo sinal com zeros
new_signal = zeros(1, N);

% Preencher o novo sinal a partir da posição 1 do novo para a posição 3 do antigo
new_signal(1:N-3) = yn3(4:end);

[num_differences, differing_positions] = compareVectors(xn, new_signal,1e-4);
plot_signal_metrics(xn, new_signal);



%% reconstrução do sinal
yn=z+w; %sem decimação
figure(5)
plot(xn); hold on
plot(yn3); 
plot(yn);  %reconstrução sem a decimação yn=2x[n-3]
title('Reconstrução do sinal')
ylabel('amplitude (mV)');
xlabel('amostras');
legend('xn','yn3','yn'); hold off
%% níveis de decomposição com a função pronta

[J,l] = wavedec(xn,3,'db2');
approx = appcoef(J,l,'db2');
[cd1,cd2,cd3] = detcoef(J,l,[1 2 3]);

%%


%saída passa alta:
figure('name','Comparativo primeira decomposição ')
plot(cd1, 'r:'); hold on
plot(zd, 'b--'); 
title('Comparativo primeira decomposição')
legend('cd1','zd'); hold off
[lpd,hpd,lpr,hpr]=wfilters('db2'); %da pela ordem os coef.
cd1m = cd1(1:end-1);
plot_signal_metrics(cd1m, zd);
[num_differences, differing_positions] = compareVectors(cd1m, zd,1e-4);

%% implementação árvore de filtros:

%segundo nível decomposição:


for i=1:1800
    if i==1
        w2(1,i) = -a*wd(1,i);
    elseif i==2
        w2(1,i) = -a*wd(1,i) - b*wd(1,i-1);
    elseif i==3
        w2(1,i) = -a*wd(1,i)- b*wd(1,i-1)- c*wd(1,i-2);
    else    
  w2(1,i) = -a*wd(1,i) - b*wd(1,i-1) - c*wd(1,i-2) - d*wd(1,i-3);
    
    
    end
end
for i=1:1800
    if i==1
       z2(1,i) = d*wd(1,i);
    elseif i==2
        z2(1,i) = d*wd(1,i)- c*wd(1,i-1);
    elseif i==3
        z2(1,i) = d*wd(1,i) - c*wd(1,i-1) + b*wd(1,i-2);
    else
        
  z2(1,i) = d*wd(1,i) - c*wd(1,i-1) + b*wd(1,i-2) - a*wd(1,i-3); 
    
    
    end
end 


%%teste com convolução: ok! 
z2conv=conv(wd,hplivro);
w2conv=conv(wd,lp);

figure('name','Comparativo convolução manual de Z2')
plot(z2); hold on
plot(z2conv); hold off
legend('z2','z2teste'); 

%%
figure('name','Comparativo convolução manual de W2')
plot(w2); hold on
plot(w2conv); hold off
legend('w2','w2conv'); 

cont=1;
for i=1:1801
  if mod(i,2)==0  
      
    w2d(1,i-cont)= w2(1,i);
    z2d(1,i-cont)= z2(1,i);
    cont= cont+1; 
  end
  
end
%%
figure('name','Comparativo segunda decomposição')
plot(cd2,'b--'); hold on
plot(z2d,'r:'); hold off
legend('cd2','z2d'); 

cd2m = cd2(1:end-2);
plot_signal_metrics(cd2m, z2d);
[num_differences, differing_positions] = compareVectors(cd2m, z2d,1e-4);

%% terceiro nível decomposição:


for i=1:900
    if i==1
        w3(1,i) = a*w2d(1,i);
    elseif i==2
        w3(1,i) = a*w2d(1,i)+ b*w2d(1,i-1);
    elseif i==3
        w3(1,i) = a*w2d(1,i)+ b*w2d(1,i-1)+ c*w2d(1,i-2);
    else    
  w3(1,i) = a*w2d(1,i)+ b*w2d(1,i-1)+ c*w2d(1,i-2)+ d*w2d(1,i-3);
    
    
    end
end
for i=1:900
    if i==1
       z3(1,i) = -d*w2d(1,i);
    elseif i==2
        z3(1,i) = -d*w2d(1,i) + c*w2d(1,i-1);
    elseif i==3
        z3(1,i) = -d*w2d(1,i) + c*w2d(1,i-1) - b*w2d(1,i-2);
    else
        
  z3(1,i) = -d*w2d(1,i) + c*w2d(1,i-1) - b*w2d(1,i-2) + a*w2d(1,i-3);
    
    
    end
end 




%% terceira dec. por 2 (análise)

cont=1;
for i=1:900
  if mod(i,2)==0  
      
    w3d(1,i-cont)= w3(1,i);
    z3d(1,i-cont)= z3(1,i);
    cont= cont+1; 
  end
  
end

figure('name','comparativo terceira decomposição hp')
plot(cd3); hold on
plot(z3d); hold off
legend('cd3','z3d'); 

%% teste com convolução:  
z3conv=conv(w2d,hp);
w3conv=conv(w2d,lp);

figure('name','z3 analise')
plot(z3); hold on
plot(z3conv); hold off
legend('z3','z3teste'); 

cd3m = cd3(1:end-2);
plot_signal_metrics(cd3m, z3d);
[num_differences, differing_positions] = compareVectors(cd3m, z3d,1e-4);

%%

figure('name','comparativo terceira decomposição para w')
plot(approx); hold on
plot(w3d); hold off
legend('Approximation Coefficients','w3d'); 

approxm = approx(1:end-2);
plot_signal_metrics(approxm, w3d);
[num_differences, differing_positions] = compareVectors(approxm, w3d,1e-4);
%% construção do vetor de dec. inteiro para envio da placa
% Organize os coeficientes de acordo com a especificação da waverec
w3dteste = w3d(:).'; % Mantendo todos os elementos
Jteste = [w3d(:).', z3d(:).', z2d(:).', zd(:).']; % Aproximação e detalhes

Nw = length(approx);
N_w = length(w3d);
% Inicializar vetor para o novo sinal com zeros
w3dnew = zeros(1, Nw);
w3dnew(1:N_w) = w3d;

% ajusta os erros para uma reconstrução efetiva
Nz3 = length(cd3);
N_tz3 = length(z3d);
% Inicializar vetor para o novo sinal com zeros
z3dnew = zeros(1, Nz3);
z3dnew(1:N_tz3) = z3d;

Nz2 = length(cd2);
N_tz2 = length(z2d);
% Inicializar vetor para o novo sinal com zeros
z2dnew = zeros(1, Nz2);
z2dnew(1:N_tz2) = z2d;

Nz1 = length(cd1);
N_tz1 = length(zd);
% Inicializar vetor para o novo sinal com zeros
zdnew = zeros(1, Nz1);
zdnew(1:N_tz1) = zd;

Jfinal = [w3dnew(:).', z3dnew(:).', z2dnew(:).', zdnew(:).']; % Aproximação e detalhes


% Lteste deve refletir os tamanhos corretos
lteste = [length(w3d), length(z3d), length(z2d), length(zd), length(xn)];

lfinal = [length(w3dnew), length(z3dnew), length(z2dnew), length(zdnew), length(xn)];

% Verifique se a soma de lteste (exceto o último) é igual ao tamanho de Jteste
if sum(lteste(1:end-1)) ~= length(Jteste)
    error('Dimensões inconsistentes entre Jteste e lteste');
end

% Reconstrução do sinal
xfteste = waverec(Jteste, lteste, 'db2');
xfinal = waverec(Jfinal, lfinal, 'db2');

figure('name','final')
title('rec')
plot(xfinal); hold on
plot(xn); hold off

% Verifique se os sinais possuem o mesmo comprimento
if length(xn) ~= length(xfinal)
    error('Os sinais têm comprimentos diferentes!');
end

% Cálculo das métricas
erro = xfinal - xn;               % Diferença entre os sinais
MAE = mean(abs(erro));             % Mean Absolute Error (Erro Absoluto Médio)
MSE = mean(erro.^2);               % Mean Squared Error (Erro Quadrático Médio)
RMSE = sqrt(MSE);                  % Root Mean Squared Error (Erro Quadrático Médio)
maxError = max(abs(erro));         % Erro Máximo
PSNR = 10 * log10(max(xn).^2 / MSE); % Peak Signal-to-Noise Ratio (PSNR)
corrCoef = corr(xn(:), xfinal(:)); % Coeficiente de Correlação

% Critérios de avaliação (baseado nos valores aceitáveis mencionados)
MAE_acceptable = MAE <= 0.05;
RMSE_acceptable = RMSE <= 0.05;
PSNR_acceptable = PSNR >= 40; % PSNR aceitável geralmente >= 40 dB
Correlation_acceptable = corrCoef >= 0.95;

% Resultado
disp('Métricas de Reconstrução do Sinal:')
fprintf('MAE: %.5f (Aceitável: %d)\n', MAE, MAE_acceptable);
fprintf('RMSE: %.5f (Aceitável: %d)\n', RMSE, RMSE_acceptable);
fprintf('Erro Máximo: %.5f\n', maxError);
fprintf('PSNR: %.2f dB (Aceitável: %d)\n', PSNR, PSNR_acceptable);
fprintf('Coeficiente de Correlação: %.5f (Aceitável: %d)\n', corrCoef, Correlation_acceptable);

% Verifica se todos os critérios são atendidos
if MAE_acceptable && RMSE_acceptable && PSNR_acceptable && Correlation_acceptable
    disp('A reconstrução do sinal atende a todos os critérios aceitáveis.');
else
    disp('A reconstrução do sinal NÃO atende a todos os critérios aceitáveis.');
end


disp('comparação xn e xfinal')
plot_signal_metrics(xn, xfinal);

disp('comparação J:')
% Comprimento do sinal original
%N2 = length(J);
%N_teste = length(Jfinal);
%new_signal2(1:N_teste) = Jfinal;
plot_signal_metrics(J, Jfinal);
[num_differences, differing_positions] = compareVectors(Jfinal, J,1e-4);

% Gráficos para visualização
figure;
subplot(2,1,1);
plot(xn, 'b', 'LineWidth', 1); hold on;
plot(xfinal, 'r--', 'LineWidth', 1.5);
title('Comparação dos Sinais');
legend('Sinal Original', 'Sinal Reconstruído');
xlabel('Amostras');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(erro, 'k', 'LineWidth', 1.5);
title('Erro (Reconstruído - Original)');
xlabel('Amostras');
ylabel('Erro');
grid on;

%%
figure('name','teste')
title('rec')
plot(xfteste); hold on
plot(xn); hold off

% Verifique se os sinais possuem o mesmo comprimento
if length(xn) ~= length(xfteste)
    error('Os sinais têm comprimentos diferentes!');
end

% Cálculo das métricas
erro = xfteste - xn;               % Diferença entre os sinais
MAE = mean(abs(erro));             % Mean Absolute Error (Erro Absoluto Médio)
MSE = mean(erro.^2);               % Mean Squared Error (Erro Quadrático Médio)
RMSE = sqrt(MSE);                  % Root Mean Squared Error (Erro Quadrático Médio)
maxError = max(abs(erro));         % Erro Máximo
PSNR = 10 * log10(max(xn).^2 / MSE); % Peak Signal-to-Noise Ratio (PSNR)
corrCoef = corr(xn(:), xfteste(:)); % Coeficiente de Correlação

% Critérios de avaliação (baseado nos valores aceitáveis mencionados)
MAE_acceptable = MAE <= 0.05;
RMSE_acceptable = RMSE <= 0.05;
PSNR_acceptable = PSNR >= 40; % PSNR aceitável geralmente >= 40 dB
Correlation_acceptable = corrCoef >= 0.95;

% Resultado
disp('Métricas de Reconstrução do Sinal:')
fprintf('MAE: %.5f (Aceitável: %d)\n', MAE, MAE_acceptable);
fprintf('RMSE: %.5f (Aceitável: %d)\n', RMSE, RMSE_acceptable);
fprintf('Erro Máximo: %.5f\n', maxError);
fprintf('PSNR: %.2f dB (Aceitável: %d)\n', PSNR, PSNR_acceptable);
fprintf('Coeficiente de Correlação: %.5f (Aceitável: %d)\n', corrCoef, Correlation_acceptable);

% Verifica se todos os critérios são atendidos
if MAE_acceptable && RMSE_acceptable && PSNR_acceptable && Correlation_acceptable
    disp('A reconstrução do sinal atende a todos os critérios aceitáveis.');
else
    disp('A reconstrução do sinal NÃO atende a todos os critérios aceitáveis.');
end


disp('comparação xn e xfteste')
plot_signal_metrics(xn, xfteste);

disp('comparação J:')
% Comprimento do sinal original
N2 = length(J);
N_teste = length(Jteste);
new_signal2(1:N_teste) = Jteste;
plot_signal_metrics(J, new_signal2);
[num_differences, differing_positions] = compareVectors(new_signal2, J,1e-4);

% Gráficos para visualização
figure;
subplot(2,1,1);
plot(xn, 'b', 'LineWidth', 1); hold on;
plot(xfteste, 'r--', 'LineWidth', 1.5);
title('Comparação dos Sinais');
legend('Sinal Original', 'Sinal Reconstruído');
xlabel('Amostras');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(erro, 'k', 'LineWidth', 1.5);
title('Erro (Reconstruído - Original)');
xlabel('Amostras');
ylabel('Erro');
grid on;

function plot_signal_metrics(s1, s2)



% Calcular o erro
    error_abs = abs(s1 - s2); % Erro absoluto
    error_sq = (s1 - s2).^2; % Erro quadrático

    % Calcular métricas
    rmse = sqrt(mean((s1 - s2).^2)); % RMSE
    prd = sqrt(sum((s1 - s2).^2) / sum(s1.^2)) * 100; % PRD

    % Exibir as métricas
    fprintf('RMSE: %.4f\n', rmse);
    fprintf('PRD: %.2f%%\n', prd);

    % Plotar os sinais e os erros
    figure;

    % Sinal original (cd1m) e reconstruído/modificado (zd)
    subplot(3, 1, 1);
    plot(s1, 'b', 'DisplayName', 'sinal1');
    hold on;
    plot(s2, 'r--', 'DisplayName', 'sinal2');
    hold off;
    legend;
    title('Sinais gerados com funções manuais e sinal waverec');
    xlabel('Tempo (s)');
    ylabel('Amplitude');
    grid on;

    % Erro absoluto
    subplot(3, 1, 2);
    plot(error_abs, 'k');
    title('Erro Absoluto |sinal1 - sinal2|');
    xlabel('Tempo (s)');
    ylabel('Erro');
    grid on;

    % Erro quadrático
    subplot(3, 1, 3);
    plot(error_sq, 'm');
    title('Erro Quadrático (sinal1 - sinal2)^2');
    xlabel('Tempo (s)');
    ylabel('Erro');
    grid on;


end


function [num_differences, differing_positions] = compareVectors(vec1, vec2, tol)
    % Define a tolerância padrão, se não for especificada
    if nargin < 3
        tol = 1e-6;
    end

    % Verifica se os vetores têm o mesmo tamanho
    if length(vec1) ~= length(vec2)
        error('Os vetores devem ter o mesmo tamanho.');
    end

    % Identifica as posições onde os valores não correspondem dentro da tolerância
    differing_positions = find(abs(vec1 - vec2) > tol);

    % Calcula o número de diferenças
    num_differences = length(differing_positions);

    % Exibe os resultados
    fprintf('Número de posições diferentes: %d\n', num_differences);
    if num_differences > 0
        fprintf('Posições diferentes: %s\n', mat2str(differing_positions));
    else
        fprintf('Os vetores são idênticos.\n');
    end
end



