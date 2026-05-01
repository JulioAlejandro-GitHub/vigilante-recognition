# Informe Técnico de Revisión del Repositorio

## 1. Resumen ejecutivo
El repositorio `vigilante-recognition` presenta un código en etapa temprana ("slice 1") enfocado en la detección de presencia humana básica. La estructura del proyecto sigue una arquitectura de diseño guiado por el dominio (DDD) limpia y separada en capas (`app/domain`, `app/infra`, `app/services`, etc.), lo cual es una excelente base. Sin embargo, por ser un bootstrap, hay áreas significativas de oportunidad, particularmente en seguridad (contraseñas por defecto, variables sensibles no protegidas), deuda técnica (uso de stubs y acoplamiento de tests a implementaciones concretas), resiliencia de la base de datos (ausencia de control de transacciones riguroso y migraciones) y observabilidad. Mejorar la configuración mediante un entorno de despliegue controlado e incluir más pruebas de integración son pasos clave.

## 2. Descripción general del sistema
El sistema es un worker ('subsistema') diseñado para procesar eventos de tipo `frame.ingested`. Al recibir estos eventos, identifica la cámara, inicializa o recupera un seguimiento humano (`human_track`), e incrementa un contador y puntaje de presencia. Basado en ciertas reglas de umbrales predefinidas en la configuración, emite decisiones sobre la presencia humana detectada, ya sea `human_presence_detected` o `human_presence_no_face`. El sistema registra un evento de reconocimiento (`recognition_event`) y encola un evento en una bandeja de salida (`event_outbox`) para ser publicado.

## 3. Arquitectura actual
La arquitectura es un worker consumidor de colas escrito en Python usando un enfoque DDD ligero:
- **Capa de Dominio (`app/domain`):** Define entidades (`FrameIngestedMessage`, `PresenceDecision`) y lógica pura para la creación de eventos (`events.py`).
- **Capa de Servicios (`app/services`):** Contiene la lógica de negocio aplicativa (`TrackService`, `PresenceService`).
- **Capa de Infraestructura (`app/infra`, `app/db.py`, `app/models.py`):** Maneja la persistencia utilizando SQLAlchemy y mapeo a PostgreSQL. Implementa el patrón Repository (`RecognitionRepository`).
- **Capa de Presentación/Consumidor (`app/worker.py`, `app/consumer.py`, `app/publisher.py`):** Actualmente el punto de entrada es el archivo `worker.py` que, junto al `consumer`, carga un fixture de prueba, inyecta dependencias manualmente, invoca los servicios, y usa un stub de publicador (`EventPublisher`) que solo loguea a consola.

## 4. Fortalezas encontradas
- **Organización Estructurada:** Clara separación de responsabilidades a través de un esquema basado en DDD (dominio, servicios, infra).
- **Tipado Fuerte:** Buen uso de sugerencias de tipos en Python (`typing`, `__future__ annotations`, Pydantic para configuraciones).
- **Abstracciones Útiles:** El patrón de Repositorio aísla la base de datos y hay un intento claro de abstracción con el outbox pattern para el desacoplo de eventos de publicación.
- **Modelos SQL Claros:** Las definiciones de SQLAlchemy en `models.py` aprovechan características modernas (tipos compuestos de PostgreSQL como JSONB y UUID) y definen esquemas de BD separados (`recognition`, `outbox`).

## 5. Puntos débiles y riesgos técnicos

| Categoría | Hallazgo | Severidad | Evidencia en archivos/rutas | Impacto | Recomendación |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Configuración | Hardcoding de credenciales por defecto | Alta | `app/config.py` | Riesgo en entornos no controlados y fuga accidental a producción. | Eliminar los defaults sensibles (`db_password`, credenciales de RabbitMQ) obligando a cargarlos desde un archivo `.env` o variables de entorno. |
| DB / Resiliencia | Ausencia de rollback / Manejo de transacciones explícito | Alta | `app/worker.py`, `app/infra/repository.py` | Posible estado inconsistente en base de datos frente a errores en servicios. | Usar context managers adecuadamente o bloques `try...except...rollback` al inyectar transacciones a los repositorios. |
| Persistencia | Múltiples `session.flush()` ineficientes / sin sentido de unidad | Media | `app/infra/repository.py` | Lento en escala; riesgo de bloqueo parcial en DB en un entorno asíncrono/concurrente. | Limitar los `flush()` o reemplazarlos por agregados que se envíen en un único `commit()` centralizado. |
| Mantenibilidad / Testing | Dependencia del worker sobre la inyección manual | Media | `app/worker.py` | Dificultad para hacer mocks en el worker o cambiar dependencias rápidamente. | Implementar un contenedor DI o encapsular el flujo principal del worker en un caso de uso (Use Case). |
| Mantenibilidad | Duplicación de lógica para pruebas en código principal | Baja | `app/worker.py` | Confusión sobre cómo es el flujo real (p.e. `repo.update_track_presence` llamado dos veces seguidas). | El worker no debería estar fuertemente acoplado a procesar "fixtures" con lógica manual. Refactorizar para uso real de RabbitMQ o desacoplar a un manejador formal. |
| Persistencia | Ausencia de migraciones de base de datos | Media | Raíz del proyecto | Imposible desplegar cambios al esquema de base de datos sin romper instancias. | Integrar `Alembic` para el control de versiones de la BD. |
| Testing | Código estricto con mocks muy acoplado | Media | `tests/test_presence_flow.py` | Las pruebas fallarán fácilmente al cambiar parámetros internos. | Usar interfaces más claras o aislar pruebas sobre valores en vez de la clase base. |

## 6. Deuda técnica
La mayor deuda técnica radica en el "Stub" en `app/publisher.py` (no implementa la conexión real a RabbitMQ), y en que el "consumidor" en `app/consumer.py` actualmente solo lee archivos locales (fixtures). Aunque esto está documentado como un paso ("primer slice"), requiere un esfuerzo considerable de rediseño de las capas exteriores (IO) para mover el proyecto a un worker asíncrono o que escuche activamente colas sin perder la estructura lógica.

Además, el archivo `app/worker.py` llama al método `repo.update_track_presence(track)` dos veces consecutivas, lo cual huele a un truco rápido para incrementar un contador para las pruebas.

## 7. Problemas de seguridad
1. **Credenciales en código (Alta):** En `app/config.py`, credenciales de base de datos (`db_user="julio"`, contraseñas en blanco) y para RabbitMQ ("guest"/"guest") son vulnerables. Obligue el uso de Pydantic BaseSettings sin valores por defecto para evitar arranques locales inseguros, o diferencie la carga de dev contra la de prod.
2. **Falta de Validación de Input (Media):** En `app/domain/entities.py`, los payloads JSON (a través del `load_fixture_message`) no validan estrictamente el esquema del evento de llegada en `consumer.py` con una herramienta moderna como Pydantic para el payload entrante; asumen que siempre contiene las claves y tipos correctos. Esto podría llevar a excepciones de `KeyError` no controladas durante la inyección.

## 8. Problemas de mantenibilidad
El ciclo de vida de la transacción en la base de datos es opaco. El `session` se crea en `app/worker.py`, se inyecta, pero cada método de repositorio hace `.flush()`. Las reglas de Clean Code sugieren dejar que el Unit of Work sea orquestado por la capa superior, sin que el repo esté llamando a `flush()` o dictando los pasos de la persistencia intermitente.
Adicionalmente, el UUID en SQLAlchemy se está especificando como `as_uuid=False`. Como Python trabaja naturalmente con el objeto `UUID`, esto forzará siempre la conversión a cadenas innecesariamente y acopla los modelos de DB fuertemente a `strings`.

## 9. Problemas de testing
- **Pruebas Limitadas:** Las pruebas existentes están bien enfocadas funcionalmente (`tests/test_presence_flow.py` valida la lógica de negocio), pero solo son un par de validaciones en humo (`smoke tests`).
- **Pruebas Faltantes:** Faltan pruebas integrales de base de datos, del comportamiento del repositorio, configuraciones y del control de errores. El patrón Outbox es crítico; se debería probar con una base en memoria o testcontainers que la persistencia y la serialización se manejen correctamente.

## 10. Problemas de performance o escalabilidad
La base de datos tiene llamados a `flush()` después de cada instrucción de creación en `app/infra/repository.py` (crear track, actualizar track). En cargas altas de cientos de eventos por segundo, el IO a la base de datos será un cuello de botella sustancial. Para ser escalable, la unidad de trabajo (Unit of Work) debe hacer un solo `commit` e inferir internamente los identificadores requeridos si se generan correctamente en la aplicación (usar uuid4 en Python en vez de generarlos en la base).

## 11. Recomendaciones técnicas priorizadas

### Prioridad alta
- Implementar validación Pydantic estricta en el consumidor entrante (`FrameIngestedMessage`).
- Configurar bloqueos `try/except` en el worker principal con un manejo estricto de transacciones de base de datos (`rollback` en caso de fallo).
- Eliminar las credenciales de base de datos y dependencias en `app/config.py` haciéndolas obligatorias mediante variables de entorno en producción.

### Prioridad media
- Integrar `Alembic` para migraciones de esquema SQL.
- Refactorizar `app/infra/repository.py` para delegar el control del flush/commit al Unit Of Work externo y optimizar I/O.
- Implementar la conexión real a RabbitMQ para reemplazar los mocks del consumidor y publicador, como siguiente avance funcional.

### Prioridad baja
- Completar la suite de pruebas mediante Pytest y fixtures parametrizados.
- Eliminar la llamada doble a `repo.update_track_presence` en el `worker.py` una vez la fuente de datos real provea secuencias coherentes.
- Configurar testcontainers para validar el I/O completo de la base de datos y la cola de mensajes en los tests de integración.

## 12. Roadmap sugerido de mejora
- **Etapa 1: estabilización** -> Remover credenciales por defecto, añadir validación Pydantic fuerte a la entrada, integrar Alembic.
- **Etapa 2: refactorización** -> Aplicar el patrón Unit of Work al `worker.py` para el control fino de las transacciones y quitar los flushes innecesarios.
- **Etapa 3: pruebas y observabilidad** -> Agregar pruebas de integración y Unit Tests de repositorios. Agregar un logger centralizado de correlación para `correlation_id`.
- **Etapa 4: optimización y escalabilidad** -> Conectar clientes verdaderos asíncronos (AIO-Pika y SQLAlchemy Async), preparando el terreno para recibir grandes volúmenes de `frame.ingested`.

## 13. Conclusión
El estado actual de `vigilante-recognition` cumple con su objetivo inicial documentado ("primer slice", bootstrap). La estructuración es madura, la lógica de dominio está bien separada de la persistencia y los modelos base son limpios. Sin embargo, antes de avanzar a integraciones reales en el "Slice 2", se deben mejorar los cimientos introduciendo validación de datos a prueba de fallos, consolidando el control de persistencia con transaccionalidad segura, eliminando secretos hardcodeados y definiendo una suite de pruebas más robusta y desacoplada.
