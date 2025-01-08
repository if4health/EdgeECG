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

% ###########################################
% Inicialização do vetor de saída
n = length(xn);
m = length(lp);
wn = zeros(1, n + m - 1); % Comprimento esperado da convolução de acordo com help wavedec
zn = zeros(1, n + m - 1);
atraso = floor((m - 1) / 2); % Calcula o atraso baseado no comprimento do filtro

% Convolução manual
for i = 1:length(wn) 
    for j = 1:m 
        if (i - j + 1 > 0) && (i - j + 1 <= n) %Índice válido no sinal 
            wn(i) = wn(i) + lp(j) * xn(i - j + 1);
        end
    end
end

% Exibição do resultado
disp('Sinal convolvido:');
disp(wn);

% Comparação com a função conv
wn_conv = conv(xn, lp);
disp('Resultado da função conv:');
disp(wn_conv);

% Verificação da equivalência
if isequal(round(wn, 10), round(wn_conv, 10)) % Arredonda para evitar erros de precisão numérica
    disp('Os resultados são equivalentes!');
else
    disp('Os resultados são diferentes!');
end

% Decimação
%wd = wn(2:2:end); % Mantém elementos em posições pares
cont=1;
for i=1:length(wn)
  if mod(i,2)==0  
      
    wd(1,i-cont)=wn(1,i);
    cont= cont+1; 
    
  end
end

% Reconstrução
wu = zeros(1, length(wd) * 2); % Inicializa vetor com zeros
wu(2:2:end) = wd; % Insere valores decimados em posições pares


zn_conv = conv(xn, hp);
for i = 1:length(zn) % Índice da saída
    for j = 1:m % Índice do filtro
        if (i - j + 1 > 0) && (i - j + 1 <= n) % Índice válido no sinal
            zn(i) = zn(i) + hp(j) * xn(i - j + 1);
        end
    end
end

% Verificação da equivalência
if isequal(round(zn, 10), round(zn_conv, 10)) % Arredonda para evitar erros de precisão numérica
    disp('Os resultados são equivalentes!');
else
    disp('Os resultados são diferentes!');
end


%Decomposição a partir do high-pass:
zd = zn(2:2:end);

figure(3)
plot(zd); hold on
legend('zd "Sinal com decimação"');

zu = zeros(1, length(zd) * 2); % Inicializa vetor com zeros
zu(2:2:end) = zd; % Insere valores decimados em posições pares
figure(4)
plot(zu);
legend('zu "Reconstrução da decimação"'); 





% Convolução manual segundo nível de dec.
nw2 = length(wd);
w2 = zeros(1, nw2 + m - 1);
for i = 1:length(w2) % Índice da saída
    for j = 1:m % Índice do filtro
        if (i - j + 1 > 0) && (i - j + 1 <= nw2) % Índice válido no sinal
            w2(i) = w2(i) + lp(j) *wd(i - j + 1);
        end
    end
end
w2_conv = conv(wd, lp);

% Decimação
w2d = w2(2:2:end); % Mantém elementos em posições pares


z2 = zeros(1, nw2 + m - 1);
for i = 1:length(w2) % Índice da saída
    for j = 1:m % Índice do filtro
        if (i - j + 1 > 0) && (i - j + 1 <= nw2) % Índice válido no sinal
            z2(i) = z2(i) + hplivro(j) *wd(i - j + 1);
        end
    end
end
z2_conv = conv(wd, hplivro);

% Decimação
z2d = z2(2:2:end); % Mantém elementos em posições pares


nw3 = length(w2d);
w3 = zeros(1, nw3 + m - 1);
%convolução do 3 nível:
for i = 1:length(w2d) % Índice da saída
    for j = 1:m % Índice do filtro
        if (i - j + 1 > 0) && (i - j + 1 <= nw3) % Índice válido no sinal
            w3(i) = w3(i) + lplivro(j) *w2d(i - j + 1);
        end
    end
end
w3_conv = conv(w2d, lp);

% Decimação
w3d = w3(2:2:end); % Mantém elementos em posições pares


nw3 = length(w2d);
z3 = zeros(1, nw3 + m - 1);
%convolução do 3 nível:
for i = 1:length(w2d) % Índice da saída
    for j = 1:m % Índice do filtro
        if (i - j + 1 > 0) && (i - j + 1 <= nw3) % Índice válido no sinal
            z3(i) = z3(i) + hp(j) *w2d(i - j + 1);
        end
    end
end
z3_conv = conv(w2d, hp);

% Decimação
z3d = z3(2:2:end); % Mantém elementos em posições pares

%para a primeira decomp:
n2 = length(wu);
wf = zeros(1, n2 + m - 1);

% Convolução manual
for i = 1:length(xn) % Índice da saída
    for j = 1:m % Índice do filtro
        if (i - j + 1 > 0) && (i - j + 1 <= n2) % Índice válido no sinal
            wf(i) = wf(i) + ilp(j) *wu(i - j + 1);
        end
    end
end

wf_conv = conv(wu, ilp);
% Verificação da equivalência
if isequal(round(wf, 10), round(wf_conv, 10)) % Arredonda para evitar erros de precisão numérica
    disp('Os resultados são equivalentes!');
else
    disp('Os resultados são diferentes!');
end


n3 = length(zu);
zf = zeros(1, n3 + m - 1);

% Convolução manual
for i = 1:length(zf) % Índice da saída
    for j = 1:m % Índice do filtro
        if (i - j + 1 > 0) && (i - j + 1 <= n3) % Índice válido no sinal
            zf(i) = zf(i) + ihp(j) *zu(i - j + 1);
        end
    end
end

zf_conv = conv(zu, ihp);
% Verificação da equivalência
if isequal(round(zf, 10), round(zf_conv, 10)) % Arredonda para evitar erros de precisão numérica
    disp('Os resultados são equivalentes!');
else
    disp('Os resultados são diferentes!');
end


figure(6)
plot(xn)
title('Signal')
[J,l] = wavedec(xn,3,'db2');
approx = appcoef(J,l,'db2');
[cd1,cd2,cd3] = detcoef(J,l,[1 2 3]);
subplot(4,1,1)
plot(approx); hold on
title('Approximation Coefficients')
subplot(4,1,2)
plot(cd3)
title('Level 3 Detail Coefficients')
subplot(4,1,3)
plot(cd2)
title('Level 2 Detail Coefficients')
subplot(4,1,4)
plot(cd1); hold off
title('Level 1 Detail Coefficients')

% construção do vetor de dec. inteiro para envio da placa


Jteste = [w3d(:).', z3d(:).', z2d(:).', zd(:).']; % Aproximação e detalhes

% Lteste deve refletir os tamanhos corretos
lteste = [length(w3d), length(z3d), length(z2d), length(zd), length(xn)];

% Verifique se a soma de lteste (exceto o último) é igual ao tamanho de Jteste
if sum(lteste(1:end-1)) ~= length(Jteste)
    error('Dimensões inconsistentes entre Jteste e lteste');
end

% Reconstrução do sinal
xfteste = waverec(Jteste, lteste, 'db2');


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

% Gráficos para visualização
figure;
subplot(2,1,1);
plot(xn, 'b', 'LineWidth', 1.5); hold on;
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

[num_differences, differing_positions] = compareVectors(Jteste, J,1e-4);

%% limear quantizado
% Determinar os índices de separação
sep1 = length(approx); % Fim da Aproximação (A3)
sep2 = sep1 + length(cd3); % Fim de D3
sep3 = sep2 + length(cd2); % Fim de D2
sep4 = sep3 + length(cd1); % Fim de D1

% Plotar os coeficientes em sequência com separadores
figure;
plot(J, 'b'); hold on;

% Adicionar linhas verticais para separar os níveis
xline(sep1, 'r--', 'A_3', 'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'right');
xline(sep2, 'r--', 'W_3', 'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'right');
xline(sep3, 'r--', 'W_2', 'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'right');
xline(sep4, 'r--', 'W_1', 'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'right');

% Configurar o gráfico
title('Representação dos Coeficientes por Escala');
xlabel('Índice');
ylabel('Amplitude');
grid on;
hold off;

%%
%níveis de quantização
%Q = 8;   % Menos níveis, maior compressão, menor qualidade
%Q = 32;  % Mais níveis, menor compressão, maior qualidade
Q=8;
% Inicialização dos valores de max e min para cd1
max_cd1 = cd1(1);
min_cd1 = cd1(1);

% Loop para calcular max e min de cd1
for i = 2:length(cd1)
    if cd1(i) > max_cd1
        max_cd1 = cd1(i);
    end
    if cd1(i) < min_cd1
        min_cd1 = cd1(i);
    end
end

% Inicialização dos valores de max e min para cd2
max_cd2 = cd2(1);
min_cd2 = cd2(1);

% Loop para calcular max e min de cd2
for i = 2:length(cd2)
    if cd2(i) > max_cd2
        max_cd2 = cd2(i);
    end
    if cd2(i) < min_cd2
        min_cd2 = cd2(i);
    end
end

% Inicialização dos valores de max e min para cd3
max_cd3 = cd3(1);
min_cd3 = cd3(1);

% Loop para calcular max e min de cd3
for i = 2:length(cd3)
    if cd3(i) > max_cd3
        max_cd3 = cd3(i);
    end
    if cd3(i) < min_cd3
        min_cd3 = cd3(i);
    end
end

% Cálculo do delta para os coeficientes
delta_cd1 = (max_cd1 - min_cd1) / Q;
delta_cd2 = (max_cd2 - min_cd2) / Q;
delta_cd3 = (max_cd3 - min_cd3) / Q;
%%
% Quantização nos coeficientes de detalhe
cd1_quant = zeros(size(cd1));
for i = 1:length(cd1)
    cd1_quant(i) = round(cd1(i) / delta_cd1) * delta_cd1;
end

cd2_quant = zeros(size(cd2));
for i = 1:length(cd2)
    cd2_quant(i) = round(cd2(i) / delta_cd2) * delta_cd2;
end

cd3_quant = zeros(size(cd3));
for i = 1:length(cd3)
    cd3_quant(i) = round(cd3(i) / delta_cd3) * delta_cd3;
end
%%

% Quantização nos coeficientes de detalhe sem usar round
cd1_quant = zeros(size(cd1));
for i = 1:length(cd1)
    % Normaliza pelo delta
    x = cd1(i) / delta_cd1;
    
    % Calcula a parte inteira e decimal
    x_int = floor(x); % Parte inteira
    x_frac = x - x_int; % Parte decimal
    
    % Implementação manual de round
    if x_frac >= 0.5
        x_quant = x_int + 1; % Arredonda para cima
    else
        x_quant = x_int; % Arredonda para baixo
    end
    
    % Retorna para a escala original
    cd1_quant(i) = x_quant * delta_cd1;
end

% Repetir para cd2
cd2_quant = zeros(size(cd2));
for i = 1:length(cd2)
    x = cd2(i) / delta_cd2;
    x_int = floor(x);
    x_frac = x - x_int;
    if x_frac >= 0.5
        x_quant = x_int + 1;
    else
        x_quant = x_int;
    end
    cd2_quant(i) = x_quant * delta_cd2;
end

% Repetir para cd3
cd3_quant = zeros(size(cd3));
for i = 1:length(cd3)
    x = cd3(i) / delta_cd3;
    x_int = floor(x);
    x_frac = x - x_int;
    if x_frac >= 0.5
        x_quant = x_int + 1;
    else
        x_quant = x_int;
    end
    cd3_quant(i) = x_quant * delta_cd3;
end



%%
% Soft threshold aplicado aos coeficientes quantizados
soft_thresh = @(x, tau) sign(x) .* max(abs(x) - tau, 0);

% Limiar adaptativo baseado no desvio padrão
tau1 = std(cd1_quant) * sqrt(2 * log(length(cd1_quant)));
tau2 = std(cd2_quant) * sqrt(2 * log(length(cd2_quant)));
tau3 = std(cd3_quant) * sqrt(2 * log(length(cd3_quant)));

cd1_thresh = soft_thresh(cd1_quant, tau1);
cd2_thresh = soft_thresh(cd2_quant, tau2);
cd3_thresh = soft_thresh(cd3_quant, tau3);

% Reconstrução do sinal com os coeficientes thresholded
J_thresh = J; % Copia os coeficientes originais para modificar
J_thresh(l(1)+1 : l(1)+l(2)) = cd3_thresh; % Substitui os coeficientes de cd3
J_thresh(l(1)+l(2)+1 : l(1)+l(2)+l(3)) = cd2_thresh; % Substitui os coeficientes de cd2
J_thresh(l(1)+l(2)+l(3)+1 : end) = cd1_thresh; % Substitui os coeficientes de cd1

% Reconstrução do sinal usando os coeficientes thresholded
x_thresh = waverec(J_thresh, l, 'db2');

% Exibir os sinais e a diferença entre eles
figure('Name', 'Comparação dos Sinais');

% Subplot 1: Sinal Original
subplot(3, 1, 1);
plot(xn, 'r');
title('Sinal Original');
xlabel('Amostras');
ylabel('Amplitude');
grid on;

% Subplot 2: Sinal Reconstruído
subplot(3, 1, 2);
plot(x_thresh, 'b');
title('Sinal Reconstruído');
xlabel('Amostras');
ylabel('Amplitude');
grid on;

% Subplot 3: Diferença entre os sinais
subplot(3, 1, 3);
plot(xn - x_thresh, 'k');
title('Diferença entre os Sinais (Erro)');
xlabel('Amostras');
ylabel('Amplitude');
grid on;

% Figura combinada para comparar diretamente os sinais
figure('Name', 'Comparação Direta dos Sinais');
plot(xn, 'r', 'LineWidth', 1.2);
hold on;
plot(x_thresh, 'b', 'LineWidth', 1.2);
title('Comparação Direta: Sinal Original e Reconstruído');
legend('Sinal Original', 'Sinal Reconstruído');
xlabel('Amostras');
ylabel('Amplitude');
grid on;
hold off;


%% Cálculo das métricas de compressão
num_coef_orig = length(J);
num_coef_nonzero = sum(J_thresh ~= 0);
CR = num_coef_orig / num_coef_nonzero; % Taxa de Compressão

PRD = 100 * sqrt(sum((xn - x_thresh).^2) / sum(xn.^2)); % Razão de Compressão

energia_total = sum(J.^2);
energia_retida = sum(J_thresh.^2);
E_retida = 100 * (energia_retida / energia_total); % Energia Retida

% Exibir métricas
fprintf('Taxa de Compressão (CR): %.2f\n', CR);
fprintf('Razão de Compressão (PRD): %.2f%%\n', PRD);
fprintf('Energia Retida: %.2f%%\n', E_retida);

% Cálculo do RMSE
RMSE = sqrt(mean((xn - x_thresh) .^ 2));
fprintf('RMSE: %.4f\n', RMSE);

% Calcular o PRD (Percentual Root-mean-square Difference)
prd = sqrt(sum((xn - x_thresh).^2) / sum(xn.^2)) * 100;

% Exibir o resultado do PRD
fprintf('PRD entre xn e x_thresh: %.2f%%\n', prd);







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












